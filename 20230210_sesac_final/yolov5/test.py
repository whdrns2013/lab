import dotenv
import os

path = '/Users/jongya/Desktop/Workspace/lab/20230210_sesac_final/yolov5/'
os.chdir(path)

# with open(path + '/aws.env', 'w') as f:
#     f.write('RDS_HOST=team06-antifragile-db.cxuncqkdvk3h.us-east-1.rds.amazonaws.com\n')
#     f.write('RDS_PORT=3306\n')
#     f.write('RDS_DATABASE=antifragile\n')
#     f.write('RDS_USERNAME=admin\n')
#     f.write('RDS_PASSWORD=antifragile1234\n')
#     f.write('S3_RESOURCE=s3\n')
#     f.write('S3_BUCKET_NAME=team06-antifragile-s3')
    
# print(dotenv.find_dotenv(path))
dotenv_path = dotenv.find_dotenv()
print(dotenv.load_dotenv(dotenv_path))
print(dotenv.load_dotenv([x for x in os.listdir(os.getcwd()) if x.endswith('.env')][0]))