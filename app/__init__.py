from flask import Flask
from flask_cors import CORS
from config import Config


def create_app(config_class=Config):
    """
    Creates and configures a Flask application instance.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app)

    with app.app_context():
        # Import and register the blueprint
        from .routes import bp as main_blueprint
        app.register_blueprint(main_blueprint)

        # You can initialize extensions here if you add any
        # e.g., db.init_app(app)

        return app
