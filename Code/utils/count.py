import os

path_where_search = "/user/cspingola/TESI/Tesi/TestSets/TestSetsV1/TestSetInternetV1/source_images"

count = 0
for elem in os.listdir(path_where_search):
    #if elem.endswith(".png"):
    count+=1
print("count", count)