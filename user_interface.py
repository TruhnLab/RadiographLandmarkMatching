
import sys
sys.path.append('./ThirdParty')

import torch
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

from utils.utils import read_csv
from measurements.measurements import get_function_dict
from romatch import roma_outdoor

class ImagePointApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Robust Medical Landmark Matching')
        self.image_size_coarse = (560,560) # horizontal,vertical
        self.image_size_upsample = (1120,1120) # horizontal,vertical
        self.image_size_show = (500,700) # horizontal,vertical
        self.knn_neighbours = 3
        
        ### Thorax
        # self.left_image_file = r'E:\data\ChestX-ray14_fullsize\preprocessed\images_001\images\00000006_000.png'
        # self.right_image_file = r'E:\data\ChestX-ray14_fullsize\preprocessed\images_001\images\00000023_000.png'
        # self.landmark_file = r'E:\data\ChestX-ray14_fullsize\preprocessed\images_001\landmarks_Alex_corrected_Eneko\00000006_000_landmarks.csv'
        # self.mode = 'thorax_pa'
        # self.landmark_scaling = (2,2)

        ### Feet Lateral Left
        # self.left_image_file = r'E:\data\UKAFeetX\annotated_lateral_Eneko\facing_left\13\13_SER_0002_1.jpg'
        # self.right_image_file = r'E:\data\UKAFeetX\annotated_lateral_Eneko\facing_left\36\36_SER_0002.jpg'
        # self.landmark_file = r'E:\data\UKAFeetX\annotated_lateral_Eneko\facing_left\13\13_landmarks.csv'
        # self.mode = 'feet_lateral'
        # self.landmark_scaling = (1,1)

        ### Feet Lateral Left
        # self.left_image_file = r'E:\data\UKAFeetX\annotated_lateral_Eneko\facing_left\13\13_SER_0002_1.jpg'
        # self.right_image_file = r'E:\data\UKAFeetX\annotated_lateral_Eneko\facing_left\36\36_SER_0002.jpg'
        # self.landmark_file = r'E:\data\UKAFeetX\annotated_lateral_Eneko\facing_left\13\13_landmarks.csv'
        # self.mode = 'feet_lateral'
        # self.landmark_scaling = (1,1)

        ### Feet AP Left extra
        # self.left_image_file = r'E:\data\UKAFeetX\annotated_AP_Alex\AP_LEFT\62\62_SER_0000.jpg'
        # self.right_image_file = r'E:\data\UKAFeetX\extra_cases\AP_LEFT\50_SER_0000.jpg'
        # self.landmark_file = r'E:\data\UKAFeetX\annotated_AP_Alex\AP_LEFT\62\62_landmarks.csv'
        # self.mode = 'feet_ap'
        # self.landmark_scaling = (1,1)

        ## Feet AP Right
        self.left_image_file = r'E:\data\UKAFeetX\annotated_ap_Eneko\Right\13\13_SER_0000.jpg'
        self.right_image_file = r'E:\data\UKAFeetX\annotated_ap_Eneko\Right\47\47_SER_0002.jpg'
        self.landmark_file = r'E:\data\UKAFeetX\annotated_ap_Eneko\Right\13\13_landmarks.csv'
        self.mode = 'feet_ap'
        self.landmark_scaling = (1,1)

        ### Knee
        # self.left_image_file = r'E:\data\UKAKneeX\9176032701-ap_lat\images\0.png'
        # self.right_image_file = r'E:\data\UKAKneeX\91922005005-right\images\0.png'
        # self.landmark_file = r'E:\data\ChestX-ray14_fullsize\preprocessed\images_001\landmarks_Alex_corrected_Eneko\00000006_000_landmarks.csv'
        # self.mode = 'thorax_pa'
        # self.landmark_scaling = (1,1)

        ### Knee Patella
        # self.left_image_file = r'E:\data\UKAPatellaX\annotated_AP_Nikol\AP_LEFT\2\2_SER_0000.jpg'
        # self.right_image_file = r'E:\data\UKAPatellaX\annotated_AP_Nikol\AP_LEFT\83\83_SER_0000.jpg'
        # self.landmark_file = r'E:\data\UKAPatellaX\annotated_AP_Nikol\AP_LEFT\2\2_landmarks.csv'
        # self.mode = 'patella'
        # self.landmark_scaling = (1,1)

        ### Shoulder
        # self.left_image_file = r'E:\data\UKAShoulderX\annotated_AP_Nikol\AP_LEFT\60\60_SER_0000.jpg'
        # self.right_image_file = r'E:\data\UKAShoulderX\annotated_AP_Nikol\AP_LEFT\1\1_SER_0000.jpg'
        # self.landmark_file = r'E:\data\UKAShoulderX\annotated_AP_Nikol\AP_LEFT\60\60_landmarks.csv'
        # self.mode = 'shoulder'
        # self.landmark_scaling = (1,1)

        # ### Testing
        # self.left_image_file = r'E:\data\UKAHipX\tests\image_s0001_i0001.jpg'
        # self.right_image_file = r'E:\data\UKAHipX\tests\3.jpg'
        # self.landmark_file = r'E:\data\UKAShoulderX\AXIAL_TEST\1\1_landmarks.csv'
        # self.mode = 'shoulder_y'
        # self.landmark_scaling = (1,1)

        self.measurement_dict, self.landmark_names = get_function_dict(self.mode) 
        self.measurement_names = [key for key in self.measurement_dict]
        self.landmark_names = ['{0}: {1}'.format(l[0], l[1]) for l in self.landmark_names]
        
        # CANVAS CONFIG

        # Load two images using Pillow
        self.left_image = Image.open(self.left_image_file)
        self.right_image = Image.open(self.right_image_file)

        # Calculate scaling for landmarks
        self.left_size_original = self.left_image.size
        self.right_size_original = self.right_image.size
        self.left_scaling = [i/j for i,j in zip(self.image_size_show, self.left_size_original)]
        self.right_scaling = [i/j for i,j in zip(self.image_size_show, self.right_size_original)]

        # Resize images if necessary
        self.left_image = self.left_image.resize(self.image_size_show)
        self.right_image = self.right_image.resize(self.image_size_show)
        
        # Convert the images to Tkinter-compatible format
        self.left_image_tk = ImageTk.PhotoImage(self.left_image)
        self.right_image_tk = ImageTk.PhotoImage(self.right_image)
        
        # Create two canvases to display the images
        self.left_canvas = tk.Canvas(root, width=self.image_size_show[0], height=self.image_size_show[1])
        self.left_canvas.grid(row=0, column=0, padx=10, pady=10)
        self.right_canvas = tk.Canvas(root, width=self.image_size_show[0], height=self.image_size_show[1])
        self.right_canvas.grid(row=0, column=1, padx=10, pady=10)
        
        # Display images on canvases
        self.left_canvas.create_image(0, 0, anchor=tk.NW, image=self.left_image_tk)
        self.right_canvas.create_image(0, 0, anchor=tk.NW, image=self.right_image_tk)
        
        # Bind click event to the left canvas
        self.left_canvas.bind("<Button-1>", self.on_left_image_click)

        # Variables to store the last points on both canvases
        self.last_left_point = None
        self.last_right_point = None



        # LANDMARK BUTTONS CONFIG

        self.landmarks = self.load_landmarks()
        self.landmarks_mapped = []
        self.drawn_landmarks = []
        self.drawn_landmarks_text = []
        self.drawn_landmarks_mapped = []
        self.drawn_landmarks_mapped_text = []
        self.drawn_measures = []
        self.text_id = None

        ## Create UI elements
        self.load_button = tk.Button(self.root, text="Load Predefined Landmarks", command=self.draw_landmarks)
        self.load_button.grid(row=1, column=0, pady=10)
        self.load_button = tk.Button(self.root, text="Erase Landmarks", command=self.erase_landmarks_all)
        self.load_button.grid(row=2, column=0, pady=10)
        self.draw_points_button = tk.Button(self.root, text="Map Landmarks", command=self.map_landmarks)
        self.draw_points_button.grid(row=1, column=1, pady=10)

        ### Create list for landmark names

        # Frame to hold the Listbox and Scrollbar
        self.list_frame = tk.Frame(self.root)
        self.list_frame.grid(row=0, column=3, padx=10, pady=10, sticky="ns")

        # Create a Listbox with a Scrollbar
        self.name_listbox = tk.Listbox(self.list_frame, width=40, height=10)
        self.name_listbox.pack(side="left", fill="y")

        # Add a vertical Scrollbar and link it to the Listbox
        self.scrollbar = tk.Scrollbar(self.list_frame, orient="vertical", command=self.name_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.name_listbox.config(yscrollcommand=self.scrollbar.set)


        # MEASUREMENT ELEMENTS CONFIG

        # Create a variable to hold the selected dropdown option
        self.dropdown_value = tk.StringVar(self.root)
        self.dropdown_value.set(self.measurement_names[0])  # Default placeholder text

        # Create and position the dropdown menu below the right canvas (right_label)
        self.dropdown_menu = tk.OptionMenu(self.root, self.dropdown_value, *self.measurement_names)
        self.dropdown_menu.grid(row=2, column=1, pady=10)  # Adjust padding for spacing

        # Call a function when an option is selected
        self.dropdown_value.trace_add("write", self.measure)



        # MATCHER CONFIG
        print('Preparing images for matching. This might take a minute...')

        # Load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        self.roma_model = roma_outdoor(device=self.device, coarse_res=self.image_size_coarse, upsample_res=self.image_size_upsample).eval()
        try: #does not work on windows
            self.roma_model = torch.compile(self.roma_model)
        except:
            pass

        with torch.no_grad():
            # Match
            warp, certainty = self.roma_model.match(self.left_image_file, self.right_image_file, device=self.device)

            # Sample matches for estimation
            matches, certainty = self.roma_model.sample(warp, certainty)
            self.kpts1, self.kpts2 = self.roma_model.to_pixel_coordinates(matches, self.image_size_show[1], self.image_size_show[0], self.image_size_show[1], self.image_size_show[0])
            self.kpts1 = self.kpts1.cpu().flip(1)
            self.kpts2 = self.kpts2.cpu().flip(1)




    def load_landmarks(self):
        landmarks = read_csv(self.landmark_file)
        landmarks = np.array([[float(y)*self.left_scaling[1]*self.landmark_scaling[1],float(x)*self.left_scaling[0]*self.landmark_scaling[0]] for x,y in landmarks]) 
        landmarks = torch.tensor(landmarks)
        return landmarks
    

    def erase_landmarks_left(self):
        for landmark, text in zip(self.drawn_landmarks, self.drawn_landmarks_text):
            self.left_canvas.delete(landmark)
            self.left_canvas.delete(text)
        self.drawn_landmarks = []
        self.drawn_landmarks_text = []


    def erase_landmarks_right(self):
        for landmark, text in zip(self.drawn_landmarks_mapped, self.drawn_landmarks_mapped_text):
            self.right_canvas.delete(landmark)
            self.right_canvas.delete(text)
        self.drawn_landmarks_mapped = []
        self.drawn_landmarks_mapped_text = []


    def erase_landmarks_individual(self):
        self.left_canvas.delete(self.last_left_point)
        self.last_left_point = None
        self.right_canvas.delete(self.last_right_point)
        self.last_right_point = None

    
    def erase_measures(self):
        for line in self.drawn_measures:
            self.right_canvas.delete(line)
        self.drawn_measures = []
        self.right_canvas.delete(self.text_id)
        self.text_id = None

    
    def erase_landmarks_all(self):
        self.erase_landmarks_left()
        self.erase_landmarks_right()
        self.erase_landmarks_individual()
        self.erase_measures()
        self.name_listbox.delete(0, tk.END)


    def draw_landmarks(self):
        self.erase_landmarks_left()
        self.name_listbox.delete(0, tk.END)
        for i, (landmark, landmark_name) in enumerate(zip(self.landmarks, self.landmark_names)):
            landmark_drawn = self.left_canvas.create_oval(int(landmark[1]) - 3, int(landmark[0]) - 3,\
                                                          int(landmark[1]) + 3, int(landmark[0]) + 3, fill="green", outline="green")
            text_drawn = self.left_canvas.create_text(int(landmark[1]), int(landmark[0]), text='{0}'.format(i+1), anchor="nw", fill="green", font=("Arial", 8))
            self.name_listbox.insert(tk.END, landmark_name)

            self.drawn_landmarks.append(landmark_drawn)
            self.drawn_landmarks_text.append(text_drawn)


    def map_landmarks(self):

        if len(self.drawn_landmarks) > 0:

            self.landmarks_mapped = []
            self.erase_landmarks_right()
            for i, landmark in enumerate(self.landmarks):
                
                # get closets grid points
                dist = torch.norm(self.kpts1-landmark, dim=1, p=None)
                knn = dist.topk(self.knn_neighbours, largest=False).indices
                closest_source_kpts = self.kpts1[knn,:]
                closest_target_kpts = self.kpts2[knn,:]

                # translate source points by average distance from source to target
                distances = closest_source_kpts - closest_target_kpts
                translation = torch.mean(distances, dim=0)
                target_kpt = landmark - translation
                self.landmarks_mapped.append(target_kpt.numpy(force=True))

                # Draw mapped landmark on the right image
                landmark_mapped = self.right_canvas.create_oval(int(target_kpt[1]) - 3, int(target_kpt[0]) - 3,\
                                                                int(target_kpt[1]) + 3, int(target_kpt[0]) + 3, fill="green", outline="green")
                text_drawn = self.right_canvas.create_text(int(target_kpt[1]), int(target_kpt[0]), text='{0}'.format(i+1), anchor="nw", fill="green", font=("Arial", 8))
                
                self.drawn_landmarks_mapped.append(landmark_mapped)
                self.drawn_landmarks_mapped_text.append(text_drawn)


    def measure(self, *args):

        if len(self.drawn_landmarks) > 0:
            self.erase_measures()

            measure_name = self.dropdown_value.get()

            # Perform measurment
            measure_function = self.measurement_dict[measure_name][0]
            measure_value = measure_function(np.array(self.landmarks_mapped))

            # Draw lines
            measure_drawing = self.measurement_dict[measure_name][1]
            measure_lines = measure_drawing(np.array(self.landmarks_mapped))[1] # take solid drawn lines

            for line in measure_lines:
                drawn_line = self.right_canvas.create_line(line[0][1], line[0][0], line[1][1], line[1][0], fill="blue", width=2) 
                self.drawn_measures.append(drawn_line)

            # Print the value on the canvas
            self.text_id = self.right_canvas.create_text(10, 10, text=f"{measure_name}:\n{measure_value:.1f}", anchor="nw", fill="white", font=("Arial", 12))


        
    def on_left_image_click(self, event):
        # Get the click coordinates on the left image
        x, y = event.x, event.y
        yx = torch.tensor([y,x])
        
        # If there's an old point on the left image, delete it
        if self.last_left_point is not None:
            self.left_canvas.delete(self.last_left_point)

        # Draw a red dot on the left image at the clicked location
        self.last_left_point = self.left_canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", outline="red")

        # get closets grid points
        dist = torch.norm(self.kpts1-yx, dim=1, p=None)
        knn = dist.topk(self.knn_neighbours, largest=False).indices
        closest_source_kpts = self.kpts1[knn,:]
        closest_target_kpts = self.kpts2[knn,:]

        # translate source points by average distance from source to target
        distances = closest_source_kpts - closest_target_kpts
        translation = torch.mean(distances, dim=0)
        target_kpt = yx - translation

        # If there's an old point on the right image, delete it
        if self.last_right_point is not None:
            self.right_canvas.delete(self.last_right_point)
        
        # Draw a red dot on the right image at the corresponding location
        self.last_right_point = self.right_canvas.create_oval(int(target_kpt[1]) - 3, int(target_kpt[0]) - 3,\
                                                              int(target_kpt[1]) + 3, int(target_kpt[0]) + 3, fill="red", outline="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePointApp(root)
    root.mainloop()
