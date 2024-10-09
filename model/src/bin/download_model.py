import os
import sys

sys.path.append(os.path.join(os.getcwd(), "../"))

from services.azure_bucket import AzureBucket

bucket = AzureBucket()
# bucket.list_blobs()
path_to_model = os.path.join(os.getcwd(), "../../processed")
bucket.download_locally("departure_arrival_model3_trained.pth", path_to_model)