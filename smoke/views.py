from django.shortcuts import redirect
from django.template import loader
from django.http import HttpResponse

def index(request):
    response = HttpResponse()
    response.write("<b>OI MEU CHAPA!!!</b>")
    return response