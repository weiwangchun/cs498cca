import boto3

class S3Utility:
    
    def __init__(self, aws_username, aws_access_key, region):
        self.s3_client = boto3.client('s3', aws_username, aws_access_key, region)
        
    def put(self, local_path, s3_bucket, s3_object_key, ExtraArgs=None):
        """
        refer:
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.upload_file
        
        :param local_path:
        :param s3_bucket:
        :param s3_object_key:
        :param ExtraArgs:
        :return:
        """
        return self.s3_client.upload_file(local_path, s3_bucket, s3_object_key, ExtraArgs = ExtraArgs)