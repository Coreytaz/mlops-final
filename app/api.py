from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from flask import Flask, request
from flask_restx import Api, Resource, fields
from pycaret.datasets import get_data
from evidently import Report
from evidently.presets import DataDriftPreset

app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False

api = Api(
    app,
    version='1.0',
    title='Simple API',
    description='A simple Flask API with Swagger documentation',
    doc='/api/docs',
    prefix='/api'
)

health_ns = api.namespace('health', description='Health check operations')

health_model = api.model('Health', {
    'status': fields.String(description='Application status')
})

drift_ns = api.namespace('drift', description='Titanic data drift simulator')

drift_request = api.model('DriftRequest', {
    'severity': fields.Float(description='0..1 intensity of the drift (default 0.35)'),
    'sample_size': fields.Integer(description='How many rows to drift inside the dataset (default 500)')
})

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / 'dataset'
REPORT_DIR = PROJECT_DIR / 'reports'
REFERENCE_PATH = DATA_DIR / 'titanic_reference.csv'
CURRENT_PATH = DATA_DIR / 'titanic_current.csv'
REPORT_PATH = REPORT_DIR / 'evidently_report.html'

def ensure_titanic_datasets() -> None:
    """Ensure Titanic reference/current datasets exist on disk (download on API startup if missing)."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_PATH.exists():
        df = get_data('titanic', verbose=False)
        df.to_csv(REFERENCE_PATH, index=False)

    if not CURRENT_PATH.exists():
        pd.read_csv(REFERENCE_PATH).to_csv(CURRENT_PATH, index=False)

try:
    ensure_titanic_datasets()
except Exception as exc:
    raise RuntimeError(f'Failed to create Titanic dataset on startup: {exc}') from exc


def apply_drift(df: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Inject controlled drift into Titanic-like data."""
    drift_df = df.copy()
    severity = max(0.0, min(float(severity), 1.0))

    rng = np.random.default_rng()

    if 'Age' in drift_df.columns:
        median_age = drift_df['Age'].median()
        noise = rng.normal(loc=10 * severity, scale=5 * severity + 1, size=len(drift_df))
        drift_df['Age'] = (drift_df['Age'].fillna(median_age) + noise).clip(lower=0)

    return drift_df


@health_ns.route('/')
class Health(Resource):
    @health_ns.doc('health_check')
    @health_ns.marshal_with(health_model)
    def get(self):
        """Health check endpoint"""
        return {'status': datetime.now() }, 200


@drift_ns.route('/simulate')
class Drift(Resource):
    @drift_ns.doc('simulate_drift')
    @drift_ns.expect(drift_request, validate=False)
    def post(self):
        """Simulate drift by generating/overwriting titanic_current.csv from titanic_reference.csv."""
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        severity = float(payload.get('severity', 0.35))
        sample_size = int(payload.get('sample_size', 500))

        ensure_titanic_datasets()
        reference_df = pd.read_csv(REFERENCE_PATH)
        if len(reference_df) == 0:
            drift_ns.abort(400, 'Reference dataset is empty, cannot simulate drift.')

        current_df = reference_df.copy()
        if sample_size and 0 < sample_size < len(current_df):
            idx = current_df.sample(n=sample_size).index
            drifted_subset = apply_drift(current_df.loc[idx].copy(), severity)
            for col in drifted_subset.columns:
                current_df.loc[idx, col] = drifted_subset[col].values
            modified_rows = int(len(idx))
        else:
            current_df = apply_drift(current_df, severity)
            modified_rows = int(len(current_df))

        current_df.to_csv(CURRENT_PATH, index=False)

        return {
            'message': 'Drift simulated and current dataset saved',
            'modified_rows': modified_rows,
            'path': str(CURRENT_PATH)
        }, 200


@drift_ns.route('/report')
class DriftReport(Resource):
    @drift_ns.doc('create_drift_report')
    def post(self):
        """Create Evidently DataDrift HTML report using titanic_reference.csv and titanic_current.csv."""

        sample_size = 500
        random_state = 42
        output_path_raw = str(REPORT_PATH)

        ensure_titanic_datasets()

        reference_df = pd.read_csv(REFERENCE_PATH)
        current_df = pd.read_csv(CURRENT_PATH)

        if len(reference_df) == 0 or len(current_df) == 0:
            drift_ns.abort(400, 'Reference/current dataset is empty, cannot build report.')

        if sample_size and sample_size > 0:
            ref_sample = reference_df.sample(n=min(sample_size, len(reference_df)), random_state=random_state)
            cur_sample = current_df.sample(n=min(sample_size, len(current_df)), random_state=random_state)
        else:
            ref_sample = reference_df
            cur_sample = current_df

        report = Report(metrics=[DataDriftPreset()])
        report = report.run(reference_data=ref_sample, current_data=cur_sample)

        output_path = Path(output_path_raw)
        if not output_path.is_absolute():
            output_path = PROJECT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report.save_html(str(output_path))

        return {
            'message': 'Data drift report generated',
            'path': str(output_path),
            'reference_rows': int(len(ref_sample)),
            'current_rows': int(len(cur_sample))
        }, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
