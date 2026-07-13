# This file defines the root route for the application. Currently, it redirects to the health check endpoint.

from flask import Blueprint, redirect

root_bp = Blueprint("root", __name__)

#TODO: Temporary redirect to health check endpoint, can be changed to a landing page in the future
@root_bp.route("/")
def root():
    return redirect("/api/health")