# Dart Recognition

This model will be able to infer the locations and field colors of up to three
darts on a dartboard.


## Dataset
### Recording dataset
1. Record images with PiCameras
2. Rename images according to `{(cam0|cam1)}_{datetime}_{token}.jpeg`
3. Annotate dart locations with the LabelMe tool
4. Create empty annotations using `add_empty_annotations()` in `data_processing/convert_labelme.py` 
5. Convert annotations to better format using `convert_labelme_json()` in `data_processing/convert_labelme.py`
6. Annotate scores using `data_processing/annotate_score.py`
