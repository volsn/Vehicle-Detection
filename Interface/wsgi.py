"""
WSGI config for Interface project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Interface.settings')

application = get_wsgi_application()


from django.http import HttpRequest
from Interface import urls

"""
from classifier import views
from Interface.urls import disactivate_all_cameras

disactivate_all_cameras()
views.start_all(HttpRequest())
print('Hello, wsgi')
"""

