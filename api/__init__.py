"""make API folder a module"""

from api.app import create_app

application = create_app()