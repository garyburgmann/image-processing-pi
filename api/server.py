from .config import API_SERVER

if API_SERVER.type == 'flask':
    from .flask_server import app as application
else:
    from .falcon_server import api as application
