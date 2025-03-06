import json
import boto3
import base64
import sagemaker
from sagemaker.serializers import IdentitySerializer

# Common setup
s3 = boto3.client('s3')
ENDPOINT = "image-classification-2023-XX-XX-XX-XX-XX"  # Replace with your endpoint
THRESHOLD = 0.93

def serializeImageData(event, context):
    """First Lambda: serialize target data from S3"""
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    s3.download_file(bucket, key, '/tmp/image.png')
    
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

def classifyImage(event, context):
    """Second Lambda: classify the image"""
    image = base64.b64decode(event['body']['image_data'])
    
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)
    predictor.serializer = IdentitySerializer("image/png")
    
    inferences = predictor.predict(image)
    
    event['body']["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': event['body']
    }

def filterInferences(event, context):
    """Third Lambda: filter low-confidence predictions"""
    inferences = json.loads(event['body']['inferences'])
    
    meets_threshold = any(x >= THRESHOLD for x in inferences)
    
    if meets_threshold:
        return {
            'statusCode': 200,
            'body': event['body']
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")