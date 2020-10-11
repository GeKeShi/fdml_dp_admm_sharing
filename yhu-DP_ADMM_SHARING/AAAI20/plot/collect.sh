#!/bin/bash

rsync -r gpu:fdml_dp_admm_sharing/result ~/Working/result/admm_result
rsync -r gpu:dpf-dml-python/result ~/Working/result/sgd_result

