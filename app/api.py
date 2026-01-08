from flask import Flask
from flask_restx import Api, Resource, fields, Namespace
from datetime import datetime

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


@health_ns.route('/')
class Health(Resource):
    @health_ns.doc('health_check')
    @health_ns.marshal_with(health_model)
    def get(self):
        """Health check endpoint"""
        return {'status': datetime.now() }, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
