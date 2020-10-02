import boto3
import json
import cv2

photo='test.jpg' #Put name of your image test file here
print('Maintaining Connection with the server........')   
client=boto3.client('rekognition',region_name='us-east-1')
print("Made Connection with the server.........")
file=open(photo,'rb').read() 

def print_details(response):
    number_of_faces=0
    for face in response["FaceDetails"]:
        if face['Confidence']>=75:
            number_of_faces+=1
            print('\n') 
            print(f"For person {number_of_faces}:")
            print('\n') 
            print(f"Age-Range:{face['AgeRange']['Low']}-{face['AgeRange']['High']}")
            print("Gender: "+face['Gender']['Value']+f" with Probability of {face['Gender']['Confidence']}%")
            print("Emotions:")
            for emotion in face["Emotions"]:
                if emotion['Confidence']>=75:
                    print(emotion['Type']+f" with Probability of {emotion['Confidence']}%")


def show_details_on_picture(response):
    image = cv2.imread(photo)
    imgWidth, imgHeight,_ = image.shape
    window_name = 'image'
    number_of_faces=0
    for face in response["FaceDetails"]:
        if face['Confidence']>=75:
            
            face_bb_left=face['BoundingBox']['Left']
            face_bb_top=face['BoundingBox']['Top']
            face_bb_width=face['BoundingBox']['Width']
            face_bb_height=face['BoundingBox']['Height']
            
            number_of_faces+=1
            
            left_point=int(imgWidth*face_bb_left)
            top_point=int(imgHeight*face_bb_top)

            abs_height=imgHeight*face_bb_height
            abs_width=imgWidth*face_bb_width

            right_point=int(left_point+abs_width)
            bottom_point=int(top_point+abs_height)

            start_point=(left_point,top_point)

            end_point=(right_point,bottom_point)
            
            image=cv2.rectangle(image,start_point ,end_point , (0,255,0), 4) 
            
    image = cv2.putText(image,"Total Faces:"+str(number_of_faces), (50, 50),cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,0),3, cv2.LINE_AA) 
    image = cv2.putText(image,"Press any key to exit", (50, 100),cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255),3, cv2.LINE_AA) 
 
    cv2.imshow(window_name, image) 
  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()                    
            

print("Waiting For Response From the Server.......")
response=client.detect_faces(
    Image={
        "Bytes":file
        },
        Attributes=["ALL"] )  

print_details(response)  
show_details_on_picture(response)      




