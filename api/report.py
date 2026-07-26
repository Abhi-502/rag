from main import handler as app_handler

# This file is used by Vercel as the Python serverless function entrypoint.
# It re-exports the top-level handler from main.py.

def handler(request):
    return app_handler(request)
