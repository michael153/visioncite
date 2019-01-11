import os
import json
import time
import random

from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, HttpRequest
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse

import images
import assets
import pipeline
from blob_cv import get_blobs_by_url

def index_view(request):
    data = {}
    bounded = pipeline.get_bounded_images()
    segments = pipeline.get_segments()

    data['ids'] = bounded
    data['blobs'] = {}
    data['rects'] = {}
    data['indexes'] = list(range(len(bounded)))
    data['cur_id'] = 0
    
    for img in bounded:
        blobs = []
        data['blobs'][img] = []
        for polygon in segments[img]:
            if len(polygon):
                minx = min([p[0] for p in polygon])
                maxx = max([p[0] for p in polygon])
                miny = min([p[1] for p in polygon])
                maxy = max([p[1] for p in polygon])
                blob = {
                    'x1': minx,
                    'x2': maxx,
                    'y1': miny,
                    'y2': maxy
                }
                data['blobs'][img].append(blob)

    return render(request, 'bounded_boxes/index.html', data)

def export_training_data_view(request):
    export_data = json.loads(request.POST['exportedValue'])
    epoch = int(time.time())
    filename = str(epoch) + "_export.json"
    filepath = 'exported_data/{0}'.format(filename)
    with open(filepath, "w") as out:
        json.dump(export_data, out, sort_keys=True, indent=4)
    with open(filepath, "rb") as f:
        response = HttpResponse(f, content_type='application/force-download')
        response['Content-Disposition'] = 'attachment; filename = {0}'.format(filename)
        response['X-Sendfile'] = filepath
        return response
    # return redirect(reverse('bounded_boxes:index'))

