import os
import requests
import tempfile
import uuid
import base64


def test_roboflow_connection(api_key, project_url):
    """Test if Roboflow connection is valid using the Python SDK approach"""
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)

        if 'roboflow.com' in project_url:
            parts = project_url.split('/')
            for i, part in enumerate(parts):
                if 'roboflow.com' in part and i + 2 < len(parts):
                    workspace_name = parts[i + 1]
                    project_name = parts[i + 2]
                    break
            else:
                return False, "Could not parse workspace and project from URL"
        else:
            return False, "Invalid Roboflow URL format"

        workspace = rf.workspace(workspace_name)
        project = workspace.project(project_name)

        return True, f"Connected to {workspace_name}/{project_name} using Python SDK"

    except ImportError:
        print("Roboflow Python SDK not available, falling back to REST API")
        return test_roboflow_connection_rest_api(api_key, project_url)

    except Exception as e:
        return False, f"Connection error: {str(e)}"


def test_roboflow_connection_rest_api(api_key, project_url):
    """Test Roboflow connection using REST API"""
    try:
        project_url = project_url.rstrip('/')

        if 'roboflow.com' in project_url:
            parts = project_url.split('/')
            for i, part in enumerate(parts):
                if 'roboflow.com' in part and i + 2 < len(parts):
                    workspace = parts[i + 1]
                    project = parts[i + 2]
                    break
            else:
                return False, "Could not parse workspace and project from URL"
        else:
            return False, "Invalid Roboflow URL format"

        test_url = f"https://api.roboflow.com/{workspace}/{project}"

        params = {'api_key': api_key}

        response = requests.get(test_url, params=params)

        if response.status_code == 200:
            return True, f"Connected to {workspace}/{project} using REST API"
        else:
            return False, f"Invalid project or API key: {response.text}"

    except Exception as e:
        return False, f"Connection error: {str(e)}"


def upload_to_roboflow_sdk(api_key, project_url, image_data, image_name, split='train', batch_name=None,
                           annotation_data=None, label_map=None):
    """Upload image to Roboflow using the Python SDK (preferred method)"""
    try:
        from roboflow import Roboflow

        if 'roboflow.com' in project_url:
            parts = project_url.split('/')
            for i, part in enumerate(parts):
                if 'roboflow.com' in part and i + 2 < len(parts):
                    workspace_name = parts[i + 1]
                    project_name = parts[i + 2]
                    break
            else:
                return False, "Could not parse workspace and project from URL"
        else:
            return False, "Invalid Roboflow URL format"

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_name).project(project_name)

        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(image_name)[0]
        unique_id = str(uuid.uuid4())[:8]
        temp_base_path = os.path.join(temp_dir, f"{base_name}-{unique_id}")

        image_temp_path = temp_base_path + '.jpg'
        annotation_temp_path = None
        labelmap_temp_path = None

        try:
            image_bytes = base64.b64decode(image_data)
            with open(image_temp_path, 'wb') as tmp_img:
                tmp_img.write(image_bytes)

            if annotation_data:
                annotation_temp_path = temp_base_path + '.txt'
                with open(annotation_temp_path, 'w') as tmp_ann:
                    tmp_ann.write(annotation_data)

                if label_map:
                    sorted_class_names = [name for id, name in sorted(label_map.items())]
                    labelmap_content = "\n".join(sorted_class_names)

                    with tempfile.NamedTemporaryFile(mode='w', suffix='.labels', delete=False) as tmp_label:
                        tmp_label.write(labelmap_content)
                        labelmap_temp_path = tmp_label.name
                    print(f"Created temporary label map: {labelmap_temp_path}")

            upload_params = {'image_path': image_temp_path, 'split': split}

            if annotation_temp_path:
                upload_params['annotation_path'] = annotation_temp_path

            if labelmap_temp_path:
                upload_params['annotation_labelmap'] = labelmap_temp_path

            if batch_name:
                upload_params['batch_name'] = batch_name

            result = project.single_upload(**upload_params)

            return True, f"Image uploaded successfully via SDK: {result}"

        finally:
            if os.path.exists(image_temp_path):
                os.remove(image_temp_path)
            if annotation_temp_path and os.path.exists(annotation_temp_path):
                os.remove(annotation_temp_path)
            if labelmap_temp_path and os.path.exists(labelmap_temp_path):
                os.remove(labelmap_temp_path)

    except ImportError:
        print("Roboflow SDK not available, falling back to REST API")
        return upload_to_roboflow_rest_api(api_key, project_url, image_data, image_name, split, batch_name,
                                           annotation_data)
    except Exception as e:
        print(f"SDK upload error: {str(e)}")
        return False, f"SDK upload error: {str(e)}"


def upload_to_roboflow_rest_api(api_key, project_url, image_data, image_name, split='train', batch_name=None,
                                annotation_data=None):
    """Upload image to Roboflow project using REST API (fallback method)"""
    try:
        project_url = project_url.rstrip('/')

        if 'roboflow.com' in project_url:
            parts = project_url.split('/')
            for i, part in enumerate(parts):
                if 'roboflow.com' in part and i + 2 < len(parts):
                    workspace = parts[i + 1]
                    project = parts[i + 2]
                    break
            else:
                return False, "Could not parse workspace and project from URL"
        else:
            return False, "Invalid Roboflow URL format"

        upload_url = f"https://api.roboflow.com/dataset/{project}/upload"

        image_bytes = base64.b64decode(image_data)

        image_temp_path = None
        annotation_temp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                image_temp_path = tmp_file.name

            files = {'file': (image_name, open(image_temp_path, 'rb'), 'image/jpeg')}

            params = {
                'api_key': api_key,
                'name': image_name,
                'split': split
            }

            if batch_name:
                params['batch'] = batch_name

            if annotation_data:
                annotation_name = os.path.splitext(image_name)[0] + '.txt'

                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as ann_file:
                    ann_file.write(annotation_data)
                    annotation_temp_path = ann_file.name

                files['annotation'] = (annotation_name, open(annotation_temp_path, 'rb'), 'text/plain')

            response = requests.post(upload_url, files=files, params=params, timeout=60)

            if response.status_code == 200:
                return True, "Image uploaded successfully via REST API"
            else:
                return False, f"Failed to upload (Status {response.status_code}): {response.text}"

        finally:
            for file_obj in files.values():
                if hasattr(file_obj[1], 'close'):
                    file_obj[1].close()

            if image_temp_path and os.path.exists(image_temp_path):
                os.remove(image_temp_path)
            if annotation_temp_path and os.path.exists(annotation_temp_path):
                os.remove(annotation_temp_path)

    except Exception as e:
        print(f"REST API upload error: {str(e)}")
        return False, f"REST API upload error: {str(e)}"


def upload_to_roboflow_api(api_key, project_url, image_data, image_name, split='train', batch_name=None,
                           annotation_data=None, label_map=None):
    """Upload image to Roboflow - tries SDK first, then falls back to REST API"""
    success, message = upload_to_roboflow_sdk(api_key, project_url, image_data, image_name, split, batch_name,
                                              annotation_data, label_map)

    if success:
        return success, message

    print(f"SDK upload failed ({message}), trying REST API...")
    return upload_to_roboflow_rest_api(api_key, project_url, image_data, image_name, split, batch_name, annotation_data)
