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
    bounded_images = pipeline.get_bounded_images()
    data['urls'] = bounded_images
    data['ids'], data['blobs'], data['rects'] = [], {}, {}
    data['indexes'] = list(range(len(bounded_images)))
    data['cur_id'] = 0
    for url in bounded_images:
        data['ids'].append(pipeline.image_dict[url])
    for url in bounded_images:
        _, blobs = get_blobs_by_url(url, pipeline.image_dict[url], useSavedData=True)
        data['blobs'][url] = []
        for i, blob in enumerate(blobs):
            x1, x2, y1, y2 = blob[0][0], blob[0][1], blob[0][2], blob[0][3]
            blob_info = {
                'x1': x1,
                'x2': x2,
                'y1': y1,
                'y2': y2,
                'words': blob[1],
                'font': blob[2]
            }
            data['blobs'][url].append(blob_info)
            data['rects'][(x1, x2, y1, y2)] = i
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

