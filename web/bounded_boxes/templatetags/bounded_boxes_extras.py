from django import template
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter
def index(l, i):
    return l[int(i)]

@register.simple_tag
def update_variable(value):
    """Allows to update existing variable in template"""
    return value