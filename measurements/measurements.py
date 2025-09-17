import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

POINT_COLOR = 'red'
LINE_COLOR = 'green'

# %%
# GENERAL FUNCTION DEFINITIONS
#---------------------------------------------------------------------
def calculate_angle(line1, line2):

    # Calculate the dot product and the magnitudes of each vector
    dot_product = np.dot(line1, line2)
    magnitude1 = np.linalg.norm(line1)
    magnitude2 = np.linalg.norm(line2)

    # Calculate the angle
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)

    return angle_deg



def perpendicular_point_on_line(line_landmark_1, line_landmark_2, landmark):
    """
    Calculate the perpendicular point from a landmark to a line defined by two landmarks.

    Parameters:
    line_landmark_1 (array-like): The first point defining the line.
    line_landmark_2 (array-like): The second point defining the line.
    landmark (array-like): The point from which the perpendicular is drawn.

    Returns:
    list: The coordinates of the perpendicular point on the line.
    """
    

    A = line_landmark_2[1] - line_landmark_1[1]
    B = line_landmark_1[0] - line_landmark_2[0]
    C = line_landmark_2[0] * line_landmark_1[1] - line_landmark_1[0] * line_landmark_2[1]

    perpendicular_landmark = [(B * (B * landmark[0] - A * landmark[1]) - A * C) / (A**2 + B**2), (-A * (B * landmark[0] - A * landmark[1]) - B * C) / (A**2 + B**2)]

    return perpendicular_landmark



def best_fit_line_from_landmarks(landmarks_subset):
    # Extract x and y
    x = np.array([pt[0] for pt in landmarks_subset])
    y = np.array([pt[1] for pt in landmarks_subset])

    # Fit line: y = mx + b
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    # Choose two x values to create two points on the line
    x0, x1 = min(x), max(x)
    point1 = (x0, m * x0 + b)
    point2 = (x1, m * x1 + b)

    return point1, point2



def plot_image(image, lines, title='', linewidth=5):

    fig, ax = plt.subplots(figsize=[10,10])
    plt.imshow(image, cmap='gray')

    #1 dots
    for num_landmark, landmark in enumerate(lines[0]):
        ax.scatter(landmark[1], landmark[0], color=lines[3][num_landmark], alpha=0.3, s=50)
        
    #2 solid line
    for num_solid, solid_line in enumerate(lines[1]):
        ax.plot([solid_line[0][1], solid_line[1][1]], [solid_line[0][0], solid_line[1][0]], '-', linewidth=linewidth, color=lines[4][num_solid])

    #3 dotted line
    for num_dotted, dotted_line in enumerate(lines[2]):
        ax.plot([dotted_line[0][1], dotted_line[1][1]], [dotted_line[0][0], dotted_line[1][0]], ':', linewidth=linewidth, color=lines[5][num_dotted])

    plt.title(title)
    plt.axis('off')



def get_function_dict(mode='feet_ap'):

    function_dict = None
    kpt_names = []
        
    if mode == 'thorax_pa':
        kpt_names = kpt_names_thorax_pa
        function_dict = {
                        'Cardiothoracic Ratio': [cardiothoracic_ratio, vis_cardiothoracic_ratio, kpts_cardiothoracic_ratio], #0
                        'Thoracic Cage Dimensionality (Rib 6)': [thoracic_cage_dim_sixth, vis_thoracic_cage_dim_sixth, kpts_thoracic_cage_dim_sixth], #1
                        'Thoracic Cage Dimensionality (Rib 9)': [thoracic_cage_dim_nineth, vis_thoracic_cage_dim_nineth, kpts_thoracic_cage_dim_nineth], #2
                        'Cardiac Apex Position': [cardiac_apex_pos, vis_cardiac_apex_pos, kpts_cardiac_apex_pos], #3
                        'Lung Height (right)': [lung_height_right, vis_lung_height_right, kpts_lung_height_right], #4
                        'Lung Height (left)': [lung_height_left, vis_lung_height_left, kpts_lung_height_left], #5
                        'Aortic Arch Width': [aortic_arch_width, vis_aortic_arch_width, kpts_aortic_arch_width], #6
                        'Clavicular Position (mid Clavicula, upper edge)': [clavicular_pos1, vis_clavicular_pos1, kpts_clavicular_pos1], #7
                        'Clavicular Position (sternal end of Clavicula)': [clavicular_pos2, vis_clavicular_pos2, kpts_clavicular_pos2], #8
                        }

    elif mode == 'feet_ap':
        kpt_names = kpt_names_feet_ap
        function_dict = {
                        'Hallux Valgus Angle': [hallux_valgus_angle, vis_hallux_valgus_angle, kpts_hallux_valgus_angle], #0
                        'First-Second Intermetatarsal Angle': [first_second_intermetatarsal_angle, vis_first_second_intermetatarsal_angle, kpts_first_second_intermetatarsal_angle], #1
                        'First-Fifth Intermetatarsal Angle': [first_fifth_intermetatarsal_angle, vis_first_fifth_intermetatarsal_angle, kpts_first_fifth_intermetatarsal_angle], #2
                        'Hallux Valgus Angle (robust)': [hallux_valgus_angle_robust, vis_hallux_valgus_angle_robust, kpts_hallux_valgus_angle_robust], #3
                        'First-Second Intermetatarsal Angle (robust)': [first_second_intermetatarsal_angle_robust, vis_first_second_intermetatarsal_angle_robust, kpts_first_second_intermetatarsal_angle_robust], #4
                        'First-Fifth Intermetatarsal Angle (robust)': [first_fifth_intermetatarsal_angle_robust, vis_first_fifth_intermetatarsal_angle_robust, kpts_first_fifth_intermetatarsal_angle_robust], #5
                        }
        
    elif mode == 'feet_lateral':
        kpt_names = kpt_names_feet_lateral
        function_dict = {
                        'Medial Arch Angle': [medial_arch_angle, vis_medial_arch_angle, kpts_medial_arch_angle], #0
                        'Talus-First Metatarsal Angle': [mearys_angle, vis_mearys_angle, kpts_mearys_angle], #1
                        'Calcaneal Inclination Angle': [calcaneal_inclination_angle, vis_calcaneal_inclination_angle, kpts_calcaneal_inclination_angle], #2
                        'Medial Arch Angle (robust)': [medial_arch_angle_robust, vis_medial_arch_angle_robust, kpts_medial_arch_angle_robust], #3
                        'Talus-First Metatarsal Angle (robust)': [mearys_angle_robust, vis_mearys_angle_robust, kpts_mearys_angle_robust], #4
                        'Calcaneal Inclination Angle (robust)': [calcaneal_inclination_angle_robust, vis_calcaneal_inclination_angle_robust, kpts_calcaneal_inclination_angle_robust], #5
                        'Calcaneal Inclination Angle v': [calcaneal_inclination_angle_v, vis_calcaneal_inclination_angle_v, kpts_calcaneal_inclination_angle_v], #6
                        }
        
    elif mode == 'shoulder_ap':
        kpt_names = kpt_names_shoulder_ap
        function_dict = {
                        'Critical Shoulder Angle': [critical_shoulder_angle, vis_critical_shoulder_angle, kpts_critical_shoulder_angle], #0
                        'Lateral Acromion Angle': [lateral_acromion_angle, vis_lateral_acromion_angle, kpts_lateral_acromion_angle], #1
                        'Acrominal Index': [acrominal_index, vis_acrominal_index, kpts_acrominal_index], #2
                        'Acromiohumeral Interval': [acromiohumeral_interval, vis_acromiohumeral_interval, kpts_acromiohumeral_interval], #3
                        'Lateral Acromion Angle (robust)': [lateral_acromion_angle_robust, vis_lateral_acromion_angle_robust, kpts_lateral_acromion_angle_robust], #4
                        }
        
    elif mode == 'shoulder_y':
        kpt_names = kpt_names_shoulder_y
        function_dict = {
                        'Acrominal Slope': [acrominal_slope, vis_acrominal_slope, kpts_acrominal_slope], #0
                        'Acrominal Tilt': [acrominal_tilt, vis_acrominal_tilt, kpts_acrominal_tilt], #1
                        'Acromiohumeral Distance': [acromiohumeral_distance, vis_acromiohumeral_distance, kpts_acromiohumeral_distance], #2
                        }
        
    elif mode == 'knee_axial':
        kpt_names = kpt_names_knee_axial
        function_dict = {
                        'Sulcus Angle': [sulcus_angle, vis_sulcus_angle, kpts_sulcus_angle], #0
                        'Congruence Angle': [congruence_angle, vis_congruence_angle, kpts_congruence_angle], #1
                        'Patella Tilting Angle': [patella_tilting_angle, vis_patella_tilting_angle, kpts_patella_tilting_angle], #2
                        'Lateral Patellofemoral Angle': [lateral_patellofemoral_angle, vis_lateral_patellofemoral_angle, kpts_lateral_patellofemoral_angle], #3
                        }
        
    elif mode == 'knee_ap':
        kpt_names = kpt_names_knee_ap
        function_dict = {
                        'Medial Joint Space': [medial_joint_space_width, vis_medial_joint_space_width, kpts_medial_joint_space_width], #0
                        'Lateral Joint Space': [lateral_joint_space_width, vis_lateral_joint_space_width, kpts_lateral_joint_space_width], #1
                        }
        
    elif mode == 'knee_lateral':
        kpt_names = kpt_names_knee_lateral
        function_dict = {
                        'Insall Salvati Ratio': [insall_salvati_ratio, vis_insall_salvati_ratio, kpts_insall_salvati_ratio], #0
                        'Modified Insall Salvati Ratio': [modified_insall_salvati_ratio, vis_modified_insall_salvati_ratio, kpts_modified_insall_salvati_ratio], #1
                        'Caton-Deschamps Index': [caton_deschamps_index, vis_caton_deschamps_index, kpts_caton_deschamps_index], #2
                        'Blackburn-Peel Ratio': [blackburn_peel_ratio, vis_blackburn_peel_ratio, kpts_blackburn_peel_ratio], #3
                        'Patella Morphology Ratio': [patella_morphology_ratio, vis_patella_morphology_ratio, kpts_patella_morphology_ratio], #4
                        'Posterior Posterior Tibial Slope': [posterior_posterior_tibial_slope, vis_posterior_posterior_tibial_slope, kpts_posterior_posterior_tibial_slope], #5
                        'Medial Posterior Tibial Slope': [medial_posterior_tibial_slope, vis_medial_posterior_tibial_slope, kpts_medial_posterior_tibial_slope], #6
                        'Anterior Posterior Tibial Slope': [anterior_posterior_tibial_slope, vis_anterior_posterior_tibial_slope, kpts_anterior_posterior_tibial_slope], #7
                        }
        
    elif mode == 'hip_ap':
        kpt_names = kpt_names_hip_ap
        function_dict = {
                        'AC Angle (Right)': [ac_angle_right, vis_ac_angle_right, kpts_ac_angle_right], #0
                        'AC Angle (Left)': [ac_angle_left, vis_ac_angle_left, kpts_ac_angle_left], #1
                        'Reimers Index (Right)': [reimers_index_right, vis_reimers_index_right, kpts_reimers_index_right], #2
                        'Reimers Index (Left)': [reimers_index_left, vis_reimers_index_left, kpts_reimers_index_left], #3
                        'Humeral Head Length (Right)': [humeral_head_length_right, vis_humeral_head_length_right, kpts_humeral_head_length_right], #4
                        'A distance (Right)': [A_right, vis_A_right, kpts_A_right], #5
                        }
    
    else: raise NotImplementedError(f"Mode '{mode}' is not implemented yet.")
    
    return function_dict, kpt_names



# %% THORAX PA MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_thorax_pa = [
    ['1', 'Costophrenic angle, right'],
    ['2', 'Right dome of diaphragm'],
    ['3', 'Angle between the right atrium and the right medial diaphragm'],
    ['4', 'Right atrium, lateral border'],
    ['5', 'Aortic arch lateral, right'],
    ['6', 'Aortic arch cranial'],
    ['7', 'Aortic arch lateral, left'],
    ['8', 'Left atrium'],
    ['9', 'Left ventricle'],
    ['10', 'Left ventricle, apex'],
    ['11', 'Left ventricle, border with the diaphragm'],
    ['12', 'Left diaphragmatic apex'],
    ['13', 'Lateral wall of the left diaphragm'],
    ['14', 'Costophrenic angle, left'],
    ['15', 'Thorax border, ninth rib, upper edge, right'],
    ['16', 'Thorax border, eighth rib, upper edge, right'],
    ['17', 'Thorax border, seventh rib, upper edge, right'],
    ['18', 'Thorax border, sixth rib, upper edge, right'],
    ['19', 'Thorax border, fifth rib, upper edge, right'],
    ['20', 'Thorax border, fourth rib, upper edge, right'],
    ['21', 'Thorax border, third rib, upper edge, right'],
    ['22', 'First rib, right, upper edge'],
    ['23', 'Lung/pleural apex/Cranial border of thorax, right'],
    ['24', 'First rib, left, upper edge'],
    ['25', 'Lung/pleural apex/Cranial border of thorax, left'],
    ['26', 'Thorax border, third rib, upper edge, left'],
    ['27', 'Thorax border, fourth rib, upper edge, left'],
    ['28', 'Thorax border, fifth rib, upper edge, left'],
    ['29', 'Thorax border, sixth rib, upper edge, left'],
    ['30', 'Thorax border, seventh rib, upper edge, left'],
    ['31', 'Thorax border, eighth rib, upper edge, left'],
    ['32', 'Thorax border, ninth rib, upper edge, left'],
    ['33', 'Middle of clavicle, upper edge, right'],
    ['34', 'Middle of clavicle, lower edge, right'],
    ['35', 'Middle of clavicle, upper edge, left'],
    ['36', 'Middle of clavicle, lower edge, left'],
    ['37', 'Sternal end of clavicle, right, cranial'],
    ['38', 'Sternal end of clavicle, right, caudal'],
    ['39', 'Sternal end of clavicle, left, cranial'],
    ['40', 'Sternal end of clavicle, left, caudal'],
    ['41', 'Humeral head, cranial border, right'],
    ['42', 'Humeral head, medial border, right'],
    ['43', 'Humeral head, caudal border, right'],
    ['44', 'Superior angle of scapula, right'],
    ['45', 'Inferior angle of scapula, right'],
    ['46', 'Lateral border of scapula, right'],
    ['47', 'Humeral head, cranial border, left'],
    ['48', 'Humeral head, medial border, left'],
    ['49', 'Humeral head, caudal border, left'],
    ['50', 'Superior angle of scapula, left'],
    ['51', 'Inferior angle of scapula, left'],
    ['52', 'Lateral border of scapula, left'],
    ['53', 'Trachea, bifurcation/carina'],
    ['54', 'Left main bronchus, upper edge'],
    ['55', 'Left main bronchus, lower edge'],
    ['56', 'Right main bronchus, upper edge'],
    ['57', 'Right main bronchus, lower edge'],
    ['58', 'Descending aorta, lateral border, left'],
    ['59', 'Second rib, upper edge, middle, right'],
    ['60', 'Third rib, upper edge, middle, right'],
    ['61', 'Fourth rib, upper edge, middle, right'],
    ['62', 'Fifth rib, upper edge, middle, right'],
    ['63', 'Sixth rib, upper edge, middle, right'],
    ['64', 'Seventh rib, upper edge, middle, right'],
    ['65', 'Eighth rib, upper edge, middle, right'],
    ['66', 'Ninth rib, upper edge, middle, right'],
    ['67', 'Second rib, upper edge, middle, left'],
    ['68', 'Third rib, upper edge, middle, left'],
    ['69', 'Fourth rib, upper edge, middle, left'],
    ['70', 'Fifth rib, upper edge, middle, left'],
    ['71', 'Sixth rib, upper edge, middle, left'],
    ['72', 'Seventh rib, upper edge, middle, left'],
    ['73', 'Eighth rib, upper edge, middle, left'],
    ['74', 'Ninth rib, upper edge, middle, left'],
    ['75', 'T1, spinous process'],
    ['76', 'T2, spinous process'],
    ['77', 'T3, spinous process'],
    ['78', 'Lower reference point - tracheal position'],
    ['79', 'Mediastinal border right'],
    ['80', 'Upper hilar border right'],
    ['81', 'Upper hilar border left']
]

#---------------------------------------------------------------------

def cardiothoracic_ratio(landmarks):
    # Ratio of the width of the heart to the width of the thorax. It is calculated by dividing the maximum horizontal
    # cardiac diameter by the maximum horizontal thoracic diameter. A CTR greater than 0.5 is generally considered 
    # abnormal and may indicate cardiomegaly.
    max_horiz_cardiac_diam = np.abs(landmarks[3][0] - landmarks[9][0])
    max_horiz_thoracic_diam = np.abs(landmarks[0][0] - landmarks[13][0])

    ctr = max_horiz_cardiac_diam/max_horiz_thoracic_diam

    return ctr

def vis_cardiothoracic_ratio(landmarks):
    # determine horizontal line
    horz1 = [[np.mean([landmarks[3][0], landmarks[9][0]]),  np.min([landmarks[3][1], landmarks[9][1]])],\
             [np.mean([landmarks[3][0], landmarks[9][0]]),  np.max([landmarks[3][1], landmarks[9][1]])]]
    horz2 = [[np.mean([landmarks[0][0], landmarks[13][0]]), np.min([landmarks[0][1], landmarks[13][1]])],\
             [np.mean([landmarks[0][0], landmarks[13][0]]), np.max([landmarks[0][1], landmarks[13][1]])]]
    return [landmarks[3], landmarks[9], landmarks[0], landmarks[13]],\
           [horz1, horz2],\
           [[landmarks[3], horz1[0]], [landmarks[9], horz1[1]], [landmarks[0], horz2[0]], [landmarks[13], horz2[1]]],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR]

def kpts_cardiothoracic_ratio():
    return [3, 9, 0, 13]

#---------------------------------------------------------------------

def thoracic_cage_dim_sixth(landmarks):
    # The maximum horizontal distance across the thorax at the 6. rip
    return np.abs(landmarks[17][0] - landmarks[28][0])

def vis_thoracic_cage_dim_sixth(landmarks):
    # determine horizontal line
    horz = [[np.mean([landmarks[17][0], landmarks[28][0]]),  np.min([landmarks[17][1], landmarks[28][1]])],\
            [np.mean([landmarks[17][0], landmarks[28][0]]),  np.max([landmarks[17][1], landmarks[28][1]])]]
    return [landmarks[17], landmarks[28]],\
           [horz],\
           [[landmarks[17], horz[0]], [landmarks[28], horz[1]]],\
           [LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR]

def kpts_thoracic_cage_dim_sixth():
    return [17, 28]

#---------------------------------------------------------------------

def thoracic_cage_dim_nineth(landmarks):
    # The maximum horizontal distance across the thorax at the 9. rip
    return np.abs(landmarks[14][0] - landmarks[31][0])

def vis_thoracic_cage_dim_nineth(landmarks):
    # determine horizontal line
    horz = [[np.mean([landmarks[14][0], landmarks[31][0]]),  np.min([landmarks[14][1], landmarks[31][1]])],\
            [np.mean([landmarks[14][0], landmarks[31][0]]),  np.max([landmarks[14][1], landmarks[31][1]])]]
    return [landmarks[14], landmarks[31]],\
           [horz],\
           [[landmarks[14], horz[0]], [landmarks[31], horz[1]]],\
           [LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR]

def kpts_thoracic_cage_dim_nineth():
    return [14, 31]

#---------------------------------------------------------------------

def cardiac_apex_pos(landmarks):
    # Location of the apex (Linker Ventrikel, Herzspitze) relative to the midline, i.e., distance (projected onto the x-axis)
    # between apex (Linker Ventrikel, Herzspitze [#10]) and mean x coordinate of 1. Brustwirbelkörper, Processus spinosus [#75],
    # 2. Brustwirbelkörper, Processus spinosus [#76], and 3. Brustwirbelkörper, Processus spinosus [#77]
    mean_spinosus_coord_y = np.mean([landmarks[74][1], landmarks[75][1], landmarks[76][1]])
    return landmarks[9][1] / mean_spinosus_coord_y

def vis_cardiac_apex_pos(landmarks):
    mean_spinosus_coord_y = np.mean([landmarks[74][1], landmarks[75][1], landmarks[76][1]])
    mean_spine_coord = (landmarks[75][0],mean_spinosus_coord_y)
    horz_line_coord = np.mean([landmarks[75][0], landmarks[9][0]])
    return [landmarks[9], landmarks[74], landmarks[75], landmarks[76], mean_spine_coord],\
           [[(horz_line_coord,mean_spinosus_coord_y),(horz_line_coord,landmarks[9][1])]],\
           [[mean_spine_coord,(horz_line_coord,mean_spinosus_coord_y)], [landmarks[9],(horz_line_coord,landmarks[9][1])],\
            [landmarks[74], mean_spine_coord], [landmarks[75], mean_spine_coord], [landmarks[76], mean_spine_coord]],\
           [LINE_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR,\
            POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_cardiac_apex_pos():
    return [9, 74, 75, 76]

#---------------------------------------------------------------------

def lung_height_right(landmarks):
    # Measurements of the lung fields’ dimensions
    return np.abs(landmarks[1][1] - landmarks[22][1])

def vis_lung_height_right(landmarks):
    horz = [[np.min([landmarks[1][0], landmarks[22][0]]), np.mean([landmarks[1][1], landmarks[22][1]])],\
            [np.max([landmarks[1][0], landmarks[22][0]]), np.mean([landmarks[1][1], landmarks[22][1]])]]
    return [landmarks[1], landmarks[22]],\
           [horz],\
           [[horz[1], landmarks[1]],[horz[0], landmarks[22]]],\
           [LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR]

def kpts_lung_height_right():
    return [1, 22]


#---------------------------------------------------------------------

def lung_height_left(landmarks):
    # Measurements of the lung fields’ dimensions
    return np.abs(landmarks[11][1] - landmarks[24][1])

def vis_lung_height_left(landmarks):
    vert = [[np.min([landmarks[11][0], landmarks[24][0]]), np.mean([landmarks[11][1], landmarks[24][1]])],\
            [np.max([landmarks[11][0], landmarks[24][0]]), np.mean([landmarks[11][1], landmarks[24][1]])]]
    return [landmarks[11], landmarks[24]],\
           [vert],\
           [[vert[1], landmarks[11]],[vert[0], landmarks[24]]],\
           [LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR]

def kpts_lung_height_left():
    return [11, 24]

#---------------------------------------------------------------------

def aortic_arch_width(landmarks):
    # The horizontal width of the aortic arch
    return np.abs(landmarks[6][0] - landmarks[4][0])

def vis_aortic_arch_width(landmarks):
    horz = [[np.mean([landmarks[6][0], landmarks[4][0]]), np.min([landmarks[6][1], landmarks[4][1]])],\
            [np.mean([landmarks[6][0], landmarks[4][0]]), np.max([landmarks[6][1], landmarks[4][1]])]]
    return [landmarks[6], landmarks[4]],\
           [horz],\
           [[horz[1], landmarks[6]],[horz[0], landmarks[4]]],\
           [LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR]

def kpts_aortic_arch_width():
    return [6, 4]

#---------------------------------------------------------------------

def clavicular_pos1(landmarks):
    # Evaluation of the clavicles for symmetry, serves to ensure the patient is not rotated and the clavicles are
    # symmetrically positioned.
    # distance (projected onto the x-axis) between the midpoints of the clavicle (Mitte der Clavicula, Oberkante, rechts [#33] or links [#35]) and the midline (as defined above)
    mean_spinosus_coord_y = np.mean([landmarks[74][1], landmarks[75][1], landmarks[76][1]])
    return np.abs(mean_spinosus_coord_y - landmarks[32][1])

def vis_clavicular_pos1(landmarks):
    mean_spinosus_coord_y = np.mean([landmarks[74][1], landmarks[75][1], landmarks[76][1]])
    mean_spine_coord = (landmarks[75][0], mean_spinosus_coord_y)
    horz_line_coord = np.mean([landmarks[75][0], landmarks[32][0]])
    return [landmarks[32], landmarks[74], landmarks[75], landmarks[76], mean_spine_coord],\
           [[(horz_line_coord,mean_spinosus_coord_y),(horz_line_coord,landmarks[32][1])]],\
           [[(landmarks[75][0],mean_spinosus_coord_y),(horz_line_coord,landmarks[75][1])], [landmarks[32],(horz_line_coord,landmarks[32][1])],\
            [landmarks[74], mean_spine_coord], [landmarks[75], mean_spine_coord], [landmarks[76], mean_spine_coord]],\
           [LINE_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR,\
            POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_clavicular_pos1():
    return [32, 74, 75, 76]

#---------------------------------------------------------------------

def clavicular_pos2(landmarks):
    # Evaluation of the clavicles for symmetry, serves to ensure the patient is not rotated and the clavicles are
    # symmetrically positioned.
    # distance (projected onto the x-axis) between the sternal ends of the clavicles (Sternales Ende der Clavicula, rechts, kranial [#37] and links [#39] and the midline (as defined above)
    mean_spinosus_coord_y = np.mean([landmarks[74][1], landmarks[75][1], landmarks[76][1]])
    return np.abs(landmarks[36][1] - mean_spinosus_coord_y)

def vis_clavicular_pos2(landmarks):
    mean_spinosus_coord_y = np.mean([landmarks[74][1], landmarks[75][1], landmarks[76][1]])
    mean_spine_coord = (landmarks[75][0], mean_spinosus_coord_y)
    horz_line_coord = np.mean([landmarks[75][0], landmarks[36][0]])
    return [landmarks[36], landmarks[74], landmarks[75], landmarks[76], mean_spine_coord],\
           [[(horz_line_coord,mean_spinosus_coord_y),(horz_line_coord,landmarks[36][1])]],\
           [[(landmarks[75][0],mean_spinosus_coord_y),(horz_line_coord,mean_spinosus_coord_y)], [landmarks[36],(horz_line_coord,landmarks[36][1])],\
            [landmarks[74], mean_spine_coord], [landmarks[75], mean_spine_coord], [landmarks[76], mean_spine_coord]],\
           [LINE_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, LINE_COLOR],\
           [LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR,\
            POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_clavicular_pos2():
    return [36, 74, 75, 76]




# %% FEET AP MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_feet_ap = [
    ['1', '1st metatarsal, base, medial border'],
    ['2', '1st metatarsal, base, lateral border'],
    ['3', '1st metatarsal, base, medial border'],
    ['4', '1st metatarsal, base, lateral border'],
    ['5', '1st metatarsal, base, medial border'],
    ['6', '1st metatarsal, base, lateral border'],
    ['7', '1st metatarsal, shaft, medial border'],
    ['8', '1st metatarsal, shaft, lateral border'],
    ['9', '1st metatarsal, shaft, medial border'],
    ['10', '1st metatarsal, shaft, lateral border'],
    ['11', '1st metatarsal, shaft, medial border'],
    ['12', '1st metatarsal, shaft, lateral border'],
    ['13', '1st metatarsal, head, medial border'],
    ['14', '1st metatarsal, head, lateral border'],
    ['15', '1st metatarsal, head, medial border'],
    ['16', '1st metatarsal, head, lateral border'],
    ['17', '1st metatarsal, head, midpoint'],
    ['18', '1st metatarsal, base, midpoint'],
    ['19', 'Proximal phalanx of 1st metatarsal, base, medial border'],
    ['20', 'Proximal phalanx of 1st metatarsal, base, lateral border'],
    ['21', 'Proximal phalanx of 1st metatarsal, base, medial border'],
    ['22', 'Proximal phalanx of 1st metatarsal, base, lateral border'],
    ['23', 'Proximal phalanx of 1st metatarsal, base, medial border'],
    ['24', 'Proximal phalanx of 1st metatarsal, base, lateral border'],
    ['25', 'Proximal phalanx of 1st metatarsal, shaft, medial border'],
    ['26', 'Proximal phalanx of 1st metatarsal, shaft, lateral border'],
    ['27', 'Proximal phalanx of 1st metatarsal, head, medial border'],
    ['28', 'Proximal phalanx of 1st metatarsal, head, lateral border'],
    ['29', 'Proximal phalanx of 1st metatarsal, head, medial border'],
    ['30', 'Proximal phalanx of 1st metatarsal, head, lateral border'],
    ['31', 'Proximal phalanx of 1st metatarsal, head, medial border'],
    ['32', 'Proximal phalanx of 1st metatarsal, head, lateral border'],
    ['33', 'Proximal phalanx of 1st metatarsal, head, midpoint'],
    ['34', 'Proximal phalanx of 1st metatarsal, base, midpoint'],
    ['35', '2nd metatarsal, base, medial border'],
    ['36', '2nd metatarsal, base, lateral border'],
    ['37', '2nd metatarsal, base, medial border'],
    ['38', '2nd metatarsal, base, lateral border'],
    ['39', '2nd metatarsal, base, medial border'],
    ['40', '2nd metatarsal, base, lateral border'],
    ['41', '2nd metatarsal, shaft, medial border'],
    ['42', '2nd metatarsal, shaft, lateral border'],
    ['43', '2nd metatarsal, shaft, medial border'],
    ['44', '2nd metatarsal, shaft, lateral border'],
    ['45', '2nd metatarsal, head, medial border'],
    ['46', '2nd metatarsal, head, lateral border'],
    ['47', '2nd metatarsal, head, medial border'],
    ['48', '2nd metatarsal, head, lateral border'],
    ['49', '2nd metatarsal, head, midpoint'],
    ['50', '2nd metatarsal, base, midpoint'],
    ['51', '5th metatarsal, base, medial border'],
    ['52', '5th metatarsal, base, lateral border'],
    ['53', '5th metatarsal, base, medial border'],
    ['54', '5th metatarsal, base, lateral border'],
    ['55', '5th metatarsal, base, medial border'],
    ['56', '5th metatarsal, base, lateral border'],
    ['57', '5th metatarsal, shaft, medial border'],
    ['58', '5th metatarsal, shaft, lateral border'],
    ['59', '5th metatarsal, shaft, medial border'],
    ['60', '5th metatarsal, shaft, lateral border'],
    ['61', '5th metatarsal, head, medial border'],
    ['62', '5th metatarsal, head, lateral border'],
    ['63', '5th metatarsal, head, medial border'],
    ['64', '5th metatarsal, head, lateral border'],
    ['65', '5th metatarsal, head, medial border'],
    ['66', '5th metatarsal, head, lateral border'],
    ['67', '5th metatarsal, head, midpoint'],
    ['68', '5th metatarsal, base, midpoint'],
]

#---------------------------------------------------------------------

def hallux_valgus_angle(landmarks):
    
    # Calculate the axes
    longitudinal_firstmetatarsal_axis = landmarks[16] - landmarks[17]
    longitudinal_phalanx_axis = landmarks[32] - landmarks[33]

    # Calculate the angle
    angle_deg = calculate_angle(longitudinal_firstmetatarsal_axis, longitudinal_phalanx_axis)
    
    return angle_deg


def vis_hallux_valgus_angle(landmarks):

    return [landmarks[16], landmarks[17], landmarks[32], landmarks[33]],\
           [[landmarks[16], landmarks[17]], [landmarks[32], landmarks[33]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_hallux_valgus_angle():
    return [16, 17, 32, 33]

#---------------------------------------------------------------------

def hallux_valgus_angle_robust(landmarks):
    
    # Calculate the axes
    longitudinal_firstmetatarsal_axis = np.mean([landmarks[16], landmarks[12], landmarks[13], landmarks[14], landmarks[15]], axis=0)\
                                      - np.mean([landmarks[17], landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5]], axis=0)
    longitudinal_phalanx_axis = np.mean([landmarks[32], landmarks[26], landmarks[27], landmarks[28], landmarks[29], landmarks[30], landmarks[31]], axis=0)\
                              - np.mean([landmarks[33], landmarks[18], landmarks[19], landmarks[20], landmarks[21], landmarks[22], landmarks[23]], axis=0)

    # Calculate the angle
    angle_deg = calculate_angle(longitudinal_firstmetatarsal_axis, longitudinal_phalanx_axis)
    
    return angle_deg


def vis_hallux_valgus_angle_robust(landmarks):

    longitudinal_firstmetatarsal_axis_point1 = np.mean([landmarks[16], landmarks[12], landmarks[13], landmarks[14], landmarks[15]], axis=0)
    longitudinal_firstmetatarsal_axis_point2 = np.mean([landmarks[17], landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5]], axis=0)
    longitudinal_phalanx_axis_point1 = np.mean([landmarks[32], landmarks[26], landmarks[27], landmarks[28], landmarks[29], landmarks[30], landmarks[31]], axis=0)
    longitudinal_phalanx_axis_point2 = np.mean([landmarks[33], landmarks[18], landmarks[19], landmarks[20], landmarks[21], landmarks[22], landmarks[23]], axis=0)

    return [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[12], landmarks[13], landmarks[14], landmarks[15],\
            landmarks[16], landmarks[17], landmarks[18], landmarks[19], landmarks[20], landmarks[21], landmarks[22], landmarks[23], landmarks[26], landmarks[27],\
            landmarks[28], landmarks[29], landmarks[30], landmarks[31], landmarks[32], landmarks[33],\
            longitudinal_firstmetatarsal_axis_point1, longitudinal_firstmetatarsal_axis_point2, longitudinal_phalanx_axis_point1, longitudinal_phalanx_axis_point2],\
           [[longitudinal_firstmetatarsal_axis_point1, longitudinal_firstmetatarsal_axis_point2], [longitudinal_phalanx_axis_point1, longitudinal_phalanx_axis_point2]],\
           [],\
           [POINT_COLOR,]*26+[LINE_COLOR,]*4,\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_hallux_valgus_angle_robust():
    return [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33]

#---------------------------------------------------------------------

def first_second_intermetatarsal_angle(landmarks):
    
    # Calculate the axes
    longitudinal_firstmetatarsal_axis = landmarks[16] - landmarks[17]
    longitudinal_secondmetatarsal_axis = landmarks[48] - landmarks[49]

    # Calculate the angle
    angle_deg = calculate_angle(longitudinal_firstmetatarsal_axis, longitudinal_secondmetatarsal_axis)
    
    return angle_deg


def vis_first_second_intermetatarsal_angle(landmarks):

    return [landmarks[16], landmarks[17], landmarks[48], landmarks[49]],\
           [[landmarks[16], landmarks[17]], [landmarks[48], landmarks[49]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_first_second_intermetatarsal_angle():
    return [16, 17, 48, 49]

#---------------------------------------------------------------------

def first_second_intermetatarsal_angle_robust(landmarks):
    
    # Calculate the axes
    longitudinal_firstmetatarsal_axis = np.mean([landmarks[16], landmarks[12], landmarks[13], landmarks[14], landmarks[15]], axis=0)\
                                      - np.mean([landmarks[17], landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5]], axis=0)
    longitudinal_secondmetatarsal_axis = np.mean([landmarks[48], landmarks[44], landmarks[45], landmarks[46], landmarks[47]], axis=0)\
                                       - np.mean([landmarks[49], landmarks[34], landmarks[35], landmarks[36], landmarks[37], landmarks[38], landmarks[39]], axis=0)

    # Calculate the angle
    angle_deg = calculate_angle(longitudinal_firstmetatarsal_axis, longitudinal_secondmetatarsal_axis)
    
    return angle_deg


def vis_first_second_intermetatarsal_angle_robust(landmarks):

    longitudinal_firstmetatarsal_axis_point1 = np.mean([landmarks[16], landmarks[12], landmarks[13], landmarks[14], landmarks[15]], axis=0)
    longitudinal_firstmetatarsal_axis_point2 = np.mean([landmarks[17], landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5]], axis=0)
    longitudinal_secondmetatarsal_axis_point1 = np.mean([landmarks[48], landmarks[44], landmarks[45], landmarks[46], landmarks[47]], axis=0)
    longitudinal_secondmetatarsal_axis_point2 = np.mean([landmarks[49], landmarks[34], landmarks[35], landmarks[36], landmarks[37], landmarks[38], landmarks[39]], axis=0)

    return [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[12], landmarks[13], landmarks[14], landmarks[15],\
            landmarks[16], landmarks[17], landmarks[34], landmarks[35], landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[44], landmarks[45],\
            landmarks[46], landmarks[47], landmarks[48], landmarks[49],\
            longitudinal_firstmetatarsal_axis_point1, longitudinal_firstmetatarsal_axis_point2, longitudinal_secondmetatarsal_axis_point1, longitudinal_secondmetatarsal_axis_point2],\
           [[longitudinal_firstmetatarsal_axis_point1, longitudinal_firstmetatarsal_axis_point2], [longitudinal_secondmetatarsal_axis_point1, longitudinal_secondmetatarsal_axis_point2]],\
           [],\
           [POINT_COLOR,]*24+[LINE_COLOR,]*4,\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_first_second_intermetatarsal_angle_robust():
    return [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49]

#---------------------------------------------------------------------

def first_fifth_intermetatarsal_angle(landmarks):
    
    # Calculate the axes
    longitudinal_firstmetatarsal_axis = landmarks[16] - landmarks[17]
    longitudinal_fifthmetatarsal_axis = landmarks[66] - landmarks[67]

    # Calculate the angle
    angle_deg = calculate_angle(longitudinal_firstmetatarsal_axis, longitudinal_fifthmetatarsal_axis)
    
    return angle_deg


def vis_first_fifth_intermetatarsal_angle(landmarks):

    return [landmarks[16], landmarks[17], landmarks[66], landmarks[67]],\
           [[landmarks[16], landmarks[17]], [landmarks[66], landmarks[67]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_first_fifth_intermetatarsal_angle():
    return [16, 17, 66, 67]

#---------------------------------------------------------------------

def first_fifth_intermetatarsal_angle_robust(landmarks):
    
    # Calculate the axes
    longitudinal_firstmetatarsal_axis = np.mean([landmarks[16], landmarks[12], landmarks[13], landmarks[14], landmarks[15]], axis=0)\
                                      - np.mean([landmarks[17], landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5]], axis=0)
    longitudinal_fifthmetatarsal_axis = np.mean([landmarks[66], landmarks[60], landmarks[61], landmarks[62], landmarks[63], landmarks[64], landmarks[65]], axis=0)\
                                      - np.mean([landmarks[67], landmarks[50], landmarks[51], landmarks[52], landmarks[53], landmarks[54], landmarks[55]], axis=0)

    # Calculate the angle
    angle_deg = calculate_angle(longitudinal_firstmetatarsal_axis, longitudinal_fifthmetatarsal_axis)
    
    return angle_deg


def vis_first_fifth_intermetatarsal_angle_robust(landmarks):

    longitudinal_firstmetatarsal_axis_point1 = np.mean([landmarks[16], landmarks[12], landmarks[13], landmarks[14], landmarks[15]], axis=0)
    longitudinal_firstmetatarsal_axis_point2 = np.mean([landmarks[17], landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5]], axis=0)
    longitudinal_fifthmetatarsal_axis_point1 = np.mean([landmarks[66], landmarks[60], landmarks[61], landmarks[62], landmarks[63], landmarks[64], landmarks[65]], axis=0)
    longitudinal_fifthmetatarsal_axis_point2 = np.mean([landmarks[67], landmarks[50], landmarks[51], landmarks[52], landmarks[53], landmarks[54], landmarks[55]], axis=0)

    return [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[12], landmarks[13], landmarks[14], landmarks[15],\
            landmarks[16], landmarks[17], landmarks[50], landmarks[51], landmarks[52], landmarks[53], landmarks[54], landmarks[55], landmarks[60],\
            landmarks[61], landmarks[62], landmarks[63], landmarks[64], landmarks[65], landmarks[66], landmarks[67],\
            longitudinal_firstmetatarsal_axis_point1, longitudinal_firstmetatarsal_axis_point2, longitudinal_fifthmetatarsal_axis_point1, longitudinal_fifthmetatarsal_axis_point2],\
           [[longitudinal_firstmetatarsal_axis_point1, longitudinal_firstmetatarsal_axis_point2], [longitudinal_fifthmetatarsal_axis_point1, longitudinal_fifthmetatarsal_axis_point2]],\
           [],\
           [POINT_COLOR,]*28+[LINE_COLOR,]*4,\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_first_fifth_intermetatarsal_angle_robust():
    return [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65, 66, 67]

#---------------------------------------------------------------------

# def talocalcaneal_angle(landmarks):
    
#     # Calculate the axes
#     talus_l_axis1 = np.mean([landmarks[38], landmarks[39]], axis=0)
#     talus_l_axis2 = np.mean([landmarks[40], landmarks[41]], axis=0)
#     talus_l_axis = talus_l_axis1 - talus_l_axis2

#     calcaneus_lat_surface = landmarks[42] - landmarks[43]

#     # Calculate the angle
#     angle_deg = calculate_angle(talus_l_axis, calcaneus_lat_surface)
    
#     return angle_deg


# def vis_talocalcaneal_angle(landmarks):
    
#     talus_l_axis1 = np.mean([landmarks[38], landmarks[39]], axis=0)
#     talus_l_axis2 = np.mean([landmarks[40], landmarks[41]], axis=0)

#     return [landmarks[38], landmarks[39], landmarks[40], landmarks[41],\
#             landmarks[42], landmarks[43], talus_l_axis1, talus_l_axis2],\
#            [[talus_l_axis1, talus_l_axis2], [landmarks[42], landmarks[43]]],\
#            [[landmarks[38], talus_l_axis1], [landmarks[39], talus_l_axis1], [landmarks[40], talus_l_axis2], [landmarks[41], talus_l_axis2]],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

# #---------------------------------------------------------------------

# def talo_first_metatarsal_angle(landmarks):
    
#     # Calculate the axes
#     hallux_l_axis1 = np.mean([landmarks[8], landmarks[9], landmarks[10], landmarks[11]], axis=0)
#     hallux_l_axis2 = np.mean([landmarks[4], landmarks[5], landmarks[6], landmarks[7]], axis=0)
#     hallux_l_axis = hallux_l_axis1 - hallux_l_axis2

#     talus_l_axis1 = np.mean([landmarks[38], landmarks[39]], axis=0)
#     talus_l_axis2 = np.mean([landmarks[40], landmarks[41]], axis=0)
#     talus_l_axis = talus_l_axis1 - talus_l_axis2

#     # Calculate the angle
#     angle_deg = calculate_angle(talus_l_axis, hallux_l_axis)
    
#     return angle_deg


# def vis_talo_first_metatarsal_angle(landmarks):
    
#     hallux_l_axis1 = np.mean([landmarks[8], landmarks[9], landmarks[10], landmarks[11]], axis=0)
#     hallux_l_axis2 = np.mean([landmarks[4], landmarks[5], landmarks[6], landmarks[7]], axis=0)

#     talus_l_axis1 = np.mean([landmarks[38], landmarks[39]], axis=0)
#     talus_l_axis2 = np.mean([landmarks[40], landmarks[41]], axis=0)

#     return [landmarks[38], landmarks[39], landmarks[40], landmarks[41],\
#             landmarks[8], landmarks[9], landmarks[10], landmarks[11],\
#             landmarks[4], landmarks[5], landmarks[6], landmarks[7],\
#             hallux_l_axis1, hallux_l_axis2, talus_l_axis1, talus_l_axis2],\
#            [[talus_l_axis1, talus_l_axis2], [hallux_l_axis1, hallux_l_axis2]],\
#            [[landmarks[8], hallux_l_axis1], [landmarks[9], hallux_l_axis1], [landmarks[10], hallux_l_axis1], [landmarks[11], hallux_l_axis1],\
#             [landmarks[4], hallux_l_axis2], [landmarks[5], hallux_l_axis2], [landmarks[6], hallux_l_axis2], [landmarks[7], hallux_l_axis2],\
#             [landmarks[38], talus_l_axis1], [landmarks[39], talus_l_axis1], [landmarks[40], talus_l_axis2], [landmarks[41], talus_l_axis2]],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

# #---------------------------------------------------------------------

# def talonavicular_coverege_angle(landmarks):
    
#     # Calculate the axes
#     navicular_articular_surface = landmarks[0] - landmarks[1]
#     talus_articular_surface = landmarks[38] - landmarks[39]
    
#     # Calculate the angle
#     angle_deg = calculate_angle(navicular_articular_surface, talus_articular_surface)
    
#     return angle_deg


# def vis_talonavicular_coverege_angle(landmarks):

#     return [landmarks[0], landmarks[1], landmarks[38], landmarks[39]],\
#            [[landmarks[0], landmarks[1]], [landmarks[38], landmarks[39]]],\
#            [],\
#            [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            []

# #---------------------------------------------------------------------

# def naviculo_metatarsal_angle(landmarks):
    
#     # Calculate the axes
#     hallux_l_axis1 = np.mean([landmarks[8], landmarks[9], landmarks[10], landmarks[11]], axis=0)
#     hallux_l_axis2 = np.mean([landmarks[4], landmarks[5], landmarks[6], landmarks[7]], axis=0)
#     hallux_l_axis = hallux_l_axis1 - hallux_l_axis2

#     navicular_articular_surface = landmarks[0] - landmarks[1]

#     # Calculate the angle
#     angle_deg = calculate_angle(navicular_articular_surface, hallux_l_axis)
    
#     return angle_deg


# def vis_naviculo_metatarsal_angle(landmarks):
    
#     hallux_l_axis1 = np.mean([landmarks[8], landmarks[9], landmarks[10], landmarks[11]], axis=0)
#     hallux_l_axis2 = np.mean([landmarks[4], landmarks[5], landmarks[6], landmarks[7]], axis=0)

#     return [landmarks[8], landmarks[9], landmarks[10], landmarks[11],\
#             landmarks[4], landmarks[5], landmarks[6], landmarks[7],\
#             hallux_l_axis1, hallux_l_axis2, landmarks[0], landmarks[1]],\
#            [[landmarks[0], landmarks[1]], [hallux_l_axis1, hallux_l_axis2]],\
#            [[landmarks[8], hallux_l_axis1], [landmarks[9], hallux_l_axis1], [landmarks[10], hallux_l_axis1], [landmarks[11], hallux_l_axis1],\
#             [landmarks[4], hallux_l_axis2], [landmarks[5], hallux_l_axis2], [landmarks[6], hallux_l_axis2], [landmarks[7], hallux_l_axis2]],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]


# %% FEET LATERAL MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_feet_lateral = [
    ['1', 'Calcaneus, tuberosity, lower edge'],
    ['2', 'Calcaneus, facet for cuboid, lower edge'],
    ['3', 'Talus, articular surface for distal tibia (dome), midpoint'],
    ['4', 'Talus, posterior process'],
    ['5', 'Talus, neck'],
    ['6', 'Talus, sulcus tali'],
    ['7', 'Talus, head, upper edge'],
    ['8', 'Talus, head, lower edge'],
    ['9', 'Talus, head, midpoint'],
    ['10', 'Talus, neck, midpoint'],
    ['11', '1st metatarsal, tarsometatarsal joint, articular surface, upper edge'],
    ['12', '1st metatarsal, tarsometatarsal joint, articular surface, lower edge'],
    ['13', '1st metatarsal, base, upper border'],
    ['14', '1st metatarsal, base, lower border'],
    ['15', '1st metatarsal, head, upper border'],
    ['16', '1st metatarsal, head, lower border'],
    ['17', '1st metatarsal, head, upper border'],
    ['18', '1st metatarsal, head, lower border'],
    ['19', '1st metatarsal, metatarsophalangeal joint, articular surface, midpoint'],
    ['20', '1st metatarsal, tarsometatarsal joint, articular surface, midpoint'],
    ['21', 'Medial sesamoid, lower border'],
    ['22', 'Medial sesamoid, lower border'],
    ['23', 'Medial sesamoid, lower border'],
    ['24', '5th metatarsal, head, lower border'],
    ['25', '5th metatarsal, head, lower border'],
    ['26', '5th metatarsal, head, lower border'],
    ['27', '5th metatarsal, head, lower border']   
]

#---------------------------------------------------------------------

def medial_arch_angle(landmarks):
    
    # Calculate the axes
    medial_arch1 = landmarks[7] - landmarks[0]
    medial_arch2 = landmarks[7] - landmarks[20]

    # Calculate the angle
    angle_deg = calculate_angle(medial_arch1, medial_arch2)
    
    return angle_deg


def vis_medial_arch_angle(landmarks):

    return [landmarks[0], landmarks[7], landmarks[20]],\
           [[landmarks[0], landmarks[7]], [landmarks[7], landmarks[20]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_medial_arch_angle():
    return [0, 7, 20]

#---------------------------------------------------------------------

def medial_arch_angle_robust(landmarks):
    
    # Calculate the axes
    medial_arch1 = landmarks[7] - landmarks[0]
    medial_arch2 = landmarks[7] - np.mean([landmarks[20], landmarks[21], landmarks[22]], axis=0)

    # Calculate the angle
    angle_deg = calculate_angle(medial_arch1, medial_arch2)
    
    return angle_deg


def vis_medial_arch_angle_robust(landmarks):

    mean_sesamoid = np.mean([landmarks[20], landmarks[21], landmarks[22]], axis=0)

    return [landmarks[0], landmarks[7], mean_sesamoid, landmarks[20], landmarks[21], landmarks[22]],\
           [[landmarks[0], landmarks[7]], [landmarks[7], mean_sesamoid]],\
           [],\
           [POINT_COLOR, POINT_COLOR, LINE_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_medial_arch_angle_robust():
    return [0, 7, 20, 21, 22]

#---------------------------------------------------------------------

def mearys_angle(landmarks):
    
    # Calculate the axes
    longitidinal_talus = landmarks[9] - landmarks[8]
    longitudinal_metatarsal = landmarks[18] - landmarks[19]

    # Calculate the angle
    angle_deg = calculate_angle(longitidinal_talus, longitudinal_metatarsal)
    
    return angle_deg


def vis_mearys_angle(landmarks):

    return [landmarks[8], landmarks[9], landmarks[18], landmarks[19]],\
           [[landmarks[8], landmarks[9]], [landmarks[18], landmarks[19]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_mearys_angle():
    return [8, 9, 18, 19]

#---------------------------------------------------------------------

def mearys_angle_robust(landmarks):
    
    # Calculate the axes
    longitidinal_talus = np.mean([landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[9]], axis=0) -\
                         np.mean([landmarks[4], landmarks[5], landmarks[6], landmarks[7], landmarks[8]], axis=0)
    longitudinal_metatarsal = np.mean([landmarks[14], landmarks[15], landmarks[16], landmarks[17], landmarks[18]], axis=0) -\
                              np.mean([landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[19]], axis=0)

    # Calculate the angle
    angle_deg = calculate_angle(longitidinal_talus, longitudinal_metatarsal)
    
    return angle_deg


def vis_mearys_angle_robust(landmarks):

    longitidinal_talus_point1 = np.mean([landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[9]], axis=0)
    longitidinal_talus_point2 = np.mean([landmarks[4], landmarks[5], landmarks[6], landmarks[7], landmarks[8]], axis=0)
    longitudinal_metatarsal_point1 = np.mean([landmarks[14], landmarks[15], landmarks[16], landmarks[17], landmarks[18]], axis=0)
    longitudinal_metatarsal_point2 = np.mean([landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[19]], axis=0)

    return [landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7],\
            landmarks[8], landmarks[9], landmarks[10], landmarks[11], landmarks[12], landmarks[13],\
            landmarks[14], landmarks[15], landmarks[16], landmarks[17], landmarks[18], landmarks[19],\
            longitidinal_talus_point1, longitidinal_talus_point2, longitudinal_metatarsal_point1, longitudinal_metatarsal_point2],\
           [[longitidinal_talus_point1, longitidinal_talus_point2], [longitudinal_metatarsal_point1, longitudinal_metatarsal_point2]],\
           [],\
           [POINT_COLOR,]*18+ [LINE_COLOR,]*4,\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_mearys_angle_robust():
    return [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

#---------------------------------------------------------------------

def calcaneal_inclination_angle(landmarks):
    
    # Calculate the axes
    support_surface = landmarks[0] - landmarks[20]
    calcaneal_inclincation_axis = landmarks[0] - landmarks[1]

    # Calculate the angle
    angle_deg = calculate_angle(calcaneal_inclincation_axis, support_surface)
    
    return angle_deg


def vis_calcaneal_inclination_angle(landmarks):

    return [landmarks[20], landmarks[1], landmarks[0]],\
           [[landmarks[20], landmarks[0]], [landmarks[1], landmarks[0]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_calcaneal_inclination_angle():
    return [20, 1, 0]

#---------------------------------------------------------------------

def calcaneal_inclination_angle_robust(landmarks):
    
    # Calculate the axes
    support_surface = landmarks[0] - np.mean([landmarks[20], landmarks[21], landmarks[22]], axis=0)
    calcaneal_inclincation_axis = landmarks[0] - landmarks[1]

    # Calculate the angle
    angle_deg = calculate_angle(calcaneal_inclincation_axis, support_surface)
    
    return angle_deg


def vis_calcaneal_inclination_angle_robust(landmarks):

    mean_sesamoid = np.mean([landmarks[20], landmarks[21], landmarks[22]], axis=0)

    return [mean_sesamoid, landmarks[1], landmarks[0], landmarks[20], landmarks[21], landmarks[22]],\
           [[mean_sesamoid, landmarks[0]], [landmarks[1], landmarks[0]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_calcaneal_inclination_angle_robust():
    return [22, 21, 20, 1, 0]

#---------------------------------------------------------------------

def calcaneal_inclination_angle_v(landmarks):
    
    # Calculate the axes
    support_surface = landmarks[0] - np.mean([landmarks[23], landmarks[24], landmarks[25], landmarks[26]], axis=0)
    calcaneal_inclincation_axis = landmarks[0] - landmarks[1]

    # Calculate the angle
    angle_deg = calculate_angle(calcaneal_inclincation_axis, support_surface)
    
    return angle_deg


def vis_calcaneal_inclination_angle_v(landmarks):

    mean_sesamoid = np.mean([landmarks[23], landmarks[24], landmarks[25], landmarks[26]], axis=0)

    return [mean_sesamoid, landmarks[1], landmarks[0], landmarks[23], landmarks[24], landmarks[25], landmarks[26]],\
           [[mean_sesamoid, landmarks[0]], [landmarks[1], landmarks[0]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_calcaneal_inclination_angle_v():
    return [26, 25, 24, 23, 1, 0]

#---------------------------------------------------------------------

# def lateral_talocalcaneal_angle(landmarks):
    
#     # Calculate the axes
#     mid_talar_axis1 = np.mean([landmarks[27], landmarks[10]], axis=0)
#     mid_talar_axis2 = np.mean([landmarks[14], landmarks[13]], axis=0)
#     mid_talar_axis = mid_talar_axis1 - mid_talar_axis2

#     calcaneal_inclincation_axis = landmarks[0] - landmarks[4]

#     # Calculate the angle
#     angle_deg = calculate_angle(calcaneal_inclincation_axis, mid_talar_axis)
    
#     return angle_deg


# def vis_lateral_talocalcaneal_angle(landmarks):
    
#     mid_talar_axis1 = np.mean([landmarks[27], landmarks[10]], axis=0)
#     mid_talar_axis2 = np.mean([landmarks[14], landmarks[13]], axis=0)

#     return [landmarks[27], landmarks[10], landmarks[14], landmarks[13],\
#             landmarks[4], landmarks[0], mid_talar_axis1, mid_talar_axis2],\
#            [[mid_talar_axis1, mid_talar_axis2], [landmarks[4], landmarks[0]]],\
#            [[landmarks[27], mid_talar_axis1], [landmarks[10], mid_talar_axis1], [landmarks[14], mid_talar_axis2], [landmarks[13], mid_talar_axis2]],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

# #---------------------------------------------------------------------

# def navicular_index(landmarks):
    
#     # Calculate the axes
#     foot_arch_length_line = landmarks[0] - landmarks[23]

#     navicular_height_perpendicular = landmarks[19] - landmarks[23]
#     projection_factor = np.dot(navicular_height_perpendicular,foot_arch_length_line) / np.dot(foot_arch_length_line,foot_arch_length_line)
#     projection_point = landmarks[23]+projection_factor*foot_arch_length_line    

#     navicular_height_line = projection_point - landmarks[19]

#     # Calculate the measurement
#     navicular_height = np.linalg.norm(navicular_height_line)
#     foot_arch_length = np.linalg.norm(foot_arch_length_line)
#     navicular_index = foot_arch_length/navicular_height
    
#     return navicular_index


# def vis_navicular_index(landmarks):
    
#     # Calculate the axes
#     foot_arch_length_line = landmarks[0] - landmarks[23]
#     navicular_height_perpendicular = landmarks[19] - landmarks[23]
#     projection_factor = np.dot(navicular_height_perpendicular,foot_arch_length_line) / np.dot(foot_arch_length_line,foot_arch_length_line)
#     projection_point = landmarks[23]+projection_factor*foot_arch_length_line

#     return [landmarks[0], landmarks[23], landmarks[19], projection_point],\
#            [[landmarks[0], landmarks[23]], [projection_point, landmarks[19]]],\
#            [],\
#            [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR, 'red'],\
#            [LINE_COLOR, LINE_COLOR],\
#            []

# #---------------------------------------------------------------------

# def calcaneal_fifth_metatarsal_angle(landmarks):
    
#     # Calculate the axes
#     calcaneal_inclincation_axis = landmarks[0] - landmarks[4]
#     inferior_edge_5_meta = landmarks[23] - landmarks[20]

#     # Calculate the angle
#     angle_deg = calculate_angle(calcaneal_inclincation_axis, inferior_edge_5_meta)
    
#     return angle_deg


# def vis_calcaneal_fifth_metatarsal_angle(landmarks):

#     return [landmarks[0], landmarks[4], landmarks[20], landmarks[23]],\
#            [[landmarks[0], landmarks[4]], [landmarks[20], landmarks[23]]],\
#            [],\
#            [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            []

# #---------------------------------------------------------------------

# def talar_declination_angle(landmarks):
    
#     # Calculate the axes
#     mid_talar_axis1 = np.mean([landmarks[27], landmarks[10]], axis=0)
#     mid_talar_axis2 = np.mean([landmarks[14], landmarks[13]], axis=0)
#     mid_talar_axis = mid_talar_axis1 - mid_talar_axis2

#     support_surface = landmarks[0] - landmarks[23]

#     # Calculate the angle
#     angle_deg = calculate_angle(support_surface, mid_talar_axis)
    
#     return angle_deg


# def vis_talar_declination_angle(landmarks):
    
#     mid_talar_axis1 = np.mean([landmarks[27], landmarks[10]], axis=0)
#     mid_talar_axis2 = np.mean([landmarks[14], landmarks[13]], axis=0)

#     return [landmarks[27], landmarks[10], landmarks[14], landmarks[13],\
#             landmarks[23], landmarks[0], mid_talar_axis1, mid_talar_axis2],\
#            [[mid_talar_axis1, mid_talar_axis2], [landmarks[23], landmarks[0]]],\
#            [[landmarks[27], mid_talar_axis1], [landmarks[10], mid_talar_axis1], [landmarks[14], mid_talar_axis2], [landmarks[13], mid_talar_axis2]],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

# #---------------------------------------------------------------------

# def first_metatarsal_declination_angle(landmarks):
    
#     # Calculate the axes
#     meta_1_l_axis1 = np.mean([landmarks[15], landmarks[16]], axis=0)
#     meta_1_l_axis2 = np.mean([landmarks[17], landmarks[18]], axis=0)
#     meta_1_l_axis = meta_1_l_axis1 - meta_1_l_axis2

#     support_surface = landmarks[23] - landmarks[0]

#     # Calculate the angle
#     angle_deg = calculate_angle(support_surface, meta_1_l_axis)
    
#     return angle_deg


# def vis_first_metatarsal_declination_angle(landmarks):
    
#     meta_1_l_axis1 = np.mean([landmarks[15], landmarks[16]], axis=0)
#     meta_1_l_axis2 = np.mean([landmarks[17], landmarks[18]], axis=0)

#     return [landmarks[15], landmarks[16], landmarks[17], landmarks[18],\
#             landmarks[23], landmarks[0], meta_1_l_axis1, meta_1_l_axis2],\
#            [[meta_1_l_axis1, meta_1_l_axis2], [landmarks[23], landmarks[0]]],\
#            [[landmarks[15], meta_1_l_axis1], [landmarks[16], meta_1_l_axis1], [landmarks[17], meta_1_l_axis2], [landmarks[18], meta_1_l_axis2]],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

# #---------------------------------------------------------------------

# def boehlers_angle(landmarks):
    
#     # Calculate the axes
#     boehler_l_1 = landmarks[7] - landmarks[9]
#     boehler_l_2 = landmarks[6] - landmarks[7]

#     # Calculate the angle
#     angle_deg = calculate_angle(boehler_l_1, boehler_l_2)
    
#     return angle_deg


# def vis_boehlers_angle(landmarks):

#     return [landmarks[6], landmarks[7], landmarks[9]],\
#            [[landmarks[6], landmarks[7]], [landmarks[7], landmarks[9]]],\
#            [],\
#            [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            []

# #---------------------------------------------------------------------

# def gisannes_angle(landmarks):
    
#     # Calculate the axes
#     gisanne_l_1 = landmarks[7] - landmarks[24]
#     gisanne_l_2 = landmarks[6] - landmarks[24]

#     # Calculate the angle
#     angle_deg = calculate_angle(gisanne_l_1, gisanne_l_2)
    
#     return angle_deg


# def vis_gisannes_angle(landmarks):

#     return [landmarks[6], landmarks[7], landmarks[24]],\
#            [[landmarks[6], landmarks[24]], [landmarks[7], landmarks[24]]],\
#            [],\
#            [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            []

# #---------------------------------------------------------------------

# def hibbs_angle(landmarks):
    
#     # Calculate the axes
#     calcaneal_inclincation_axis = landmarks[0] - landmarks[4]

#     meta_1_l_axis1 = np.mean([landmarks[15], landmarks[16]], axis=0)
#     meta_1_l_axis2 = np.mean([landmarks[17], landmarks[18]], axis=0)
#     meta_1_l_axis = meta_1_l_axis1 - meta_1_l_axis2

#     # Calculate the angle
#     angle_deg = calculate_angle(calcaneal_inclincation_axis, meta_1_l_axis)
    
#     return angle_deg


# def vis_hibbs_angle(landmarks):

#     meta_1_l_axis1 = np.mean([landmarks[15], landmarks[16]], axis=0)
#     meta_1_l_axis2 = np.mean([landmarks[17], landmarks[18]], axis=0)

#     return [landmarks[15], landmarks[16], landmarks[17], landmarks[18],\
#             landmarks[0], landmarks[4], meta_1_l_axis1, meta_1_l_axis2],\
#            [[landmarks[0], landmarks[4]], [meta_1_l_axis1, meta_1_l_axis2]],\
#            [[landmarks[15], meta_1_l_axis1], [landmarks[16], meta_1_l_axis1], [landmarks[17], meta_1_l_axis2], [landmarks[18], meta_1_l_axis2]],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR,\
#             LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
#            [LINE_COLOR, LINE_COLOR],\
#            [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]



# %% SHOULDER AP MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_shoulder_ap = [
    ['1', 'Superior lateral edge of the greater tuberosity of the humeral head'],
    ['2', 'Lateral border of the greater tuberosity of the humeral head'],
    ['3', 'Lateral border of the surgical neck of the humerus'],
    ['4', 'Superior border of the glenoid cavity'],
    ['5', 'Inferior border of the glenoid cavity'],
    ['6', 'Inferior lateral edge of the acromion'],
    ['7', 'Inferior medial edge of the acromion'],
    ['8', 'Equidistant point 1 along undersurface of the acromion'],
    ['9', 'Equidistant point 2 along undersurface of the acromion'],
    ['10', 'Equidistant point 3 along undersurface of the acromion'],
    ['11', 'Equidistant point 4 along undersurface of the acromion'],
    ['12', 'Equidistant point 5 along undersurface of the acromion'],
    ['13', 'Equidistant point 1 along upper surface of the humeral head'],
    ['14', 'Equidistant point 2 along upper surface of the humeral head'],
    ['15', 'Equidistant point 3 along upper surface of the humeral head'],
    ['16', 'Equidistant point 4 along upper surface of the humeral head'],
    ['17', 'Equidistant point 5 along upper surface of the humeral head'],
    ['18', 'Equidistant point 6 along upper surface of the humeral head'],
    ['19', 'Equidistant point 7 along upper surface of the humeral head'],
    ['20', 'Equidistant point 8 along upper surface of the humeral head']
]

#---------------------------------------------------------------------

def critical_shoulder_angle(landmarks):
    
    # Calculate the axes
    glenoid_inf_sup = landmarks[3] - landmarks[4]
    glen_to_acromion = landmarks[5] - landmarks[4]

    # Calculate the angle
    angle_deg = calculate_angle(glenoid_inf_sup, glen_to_acromion)
    
    return angle_deg


def vis_critical_shoulder_angle(landmarks):

    return [landmarks[3], landmarks[4], landmarks[5]],\
           [[landmarks[3], landmarks[4]], [landmarks[5], landmarks[4]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_critical_shoulder_angle():
    return [3, 4, 5]

#---------------------------------------------------------------------

def lateral_acromion_angle(landmarks):
    
    # Calculate the axes
    acromion_inferior_border = landmarks[6] - landmarks[5]
    glenoid_inf_sup = landmarks[3] - landmarks[4]

    # Calculate the angle
    angle_deg = calculate_angle(glenoid_inf_sup, acromion_inferior_border)
    
    return angle_deg


def vis_lateral_acromion_angle(landmarks):

    return [landmarks[3], landmarks[4], landmarks[5], landmarks[6]],\
           [[landmarks[3], landmarks[4]], [landmarks[6], landmarks[5]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_lateral_acromion_angle():
    return [3, 4, 5, 6]

#---------------------------------------------------------------------

def lateral_acromion_angle_robust(landmarks):
    
    acromion_landmarks = np.array([landmarks[6], landmarks[11], landmarks[10], landmarks[9], landmarks[8], landmarks[7], landmarks[5]])
    acromion_direction_vectors = []
    # Only calculate one-directional vectors (i < j)
    for i in range(len(acromion_landmarks)):
        for j in range(i + 1, len(acromion_landmarks)):
            acromion_direction_vectors.append(acromion_landmarks[i] - acromion_landmarks[j])
    
    # Create the mean direction vector for the acromion inferior border
    acromion_inferior_border = np.mean(acromion_direction_vectors, axis=0) 
    
    # Use the original landmarks for glenoid axis
    glenoid_inf_sup = landmarks[3] - landmarks[4]

    # Calculate the angle
    angle_deg = calculate_angle(glenoid_inf_sup, acromion_inferior_border)
    
    return angle_deg


def vis_lateral_acromion_angle_robust(landmarks):
    
    # Get acromion landmarks
    acromion_landmarks = np.array([landmarks[5], landmarks[6], landmarks[7], landmarks[8], landmarks[9], landmarks[10], landmarks[11]])
    
    # Fit a line
    x = acromion_landmarks[:, 0]
    y = acromion_landmarks[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Create two points on the fitted line for visualization
    x_min = np.min(x)
    x_max = np.max(x)
    line_point1 = np.array([x_min, m * x_min + c])
    line_point2 = np.array([x_max, m * x_max + c])
    
    # Get all landmarks to display
    display_landmarks = [landmarks[3], landmarks[4]] + [landmarks[i] for i in range(5, 12)]
    
    # Create display colors
    landmark_colors = [POINT_COLOR, POINT_COLOR] + [POINT_COLOR] * 7
    
    return display_landmarks,\
           [[landmarks[3], landmarks[4]], [line_point1, line_point2]],\
           [],\
           landmark_colors,\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_lateral_acromion_angle_robust():
    return [3, 4, 5, 6, 7, 8, 9, 10, 11]

#---------------------------------------------------------------------

def acrominal_index(landmarks):
    
    # For landmarks 1-3 determine the perpendicular point on the line between landmarks 4 and 5
    perpendicular1 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    perpendicular_dist1 = np.sqrt((perpendicular1[0]-landmarks[0][0])**2 + (perpendicular1[1]-landmarks[0][1])**2)

    perpendicular2 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perpendicular_dist2 = np.sqrt((perpendicular2[0]-landmarks[1][0])**2 + (perpendicular2[1]-landmarks[1][1])**2)
    
    perpendicular3 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[2])
    perpendicular_dist3 = np.sqrt((perpendicular3[0]-landmarks[2][0])**2 + (perpendicular3[1]-landmarks[2][1])**2)

    glenoid_length_humerus_head = np.max([perpendicular_dist1, perpendicular_dist2, perpendicular_dist3])

    # For landmark 6 determine the perpendicular point on the line between landmarks 4 and 5
    perpendicular6 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[5])
    glenoid_length_acromion = np.sqrt((perpendicular6[0]-landmarks[5][0])**2 + (perpendicular6[1]-landmarks[5][1])**2)

    return glenoid_length_acromion/glenoid_length_humerus_head


def vis_acrominal_index(landmarks):

    # For landmarks 1-3 determine the perpendicular point on the line between landmarks 4 and 5
    perpendicular1 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    perpendicular_dist1 = np.sqrt((perpendicular1[0]-landmarks[0][0])**2 + (perpendicular1[1]-landmarks[0][1])**2)

    perpendicular2 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perpendicular_dist2 = np.sqrt((perpendicular2[0]-landmarks[1][0])**2 + (perpendicular2[1]-landmarks[1][1])**2)
    
    perpendicular3 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[2])
    perpendicular_dist3 = np.sqrt((perpendicular3[0]-landmarks[2][0])**2 + (perpendicular3[1]-landmarks[2][1])**2)

    perpendicular123 = [perpendicular1,perpendicular2,perpendicular3][np.argmax([perpendicular_dist1, perpendicular_dist2, perpendicular_dist3])]
    chosen_landmark = [landmarks[0],landmarks[1],landmarks[2]][np.argmax([perpendicular_dist1, perpendicular_dist2, perpendicular_dist3])]

    # For landmark 6 determine the perpendicular point on the line between landmarks 4 and 5
    perpendicular6 = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[5])

    return [chosen_landmark, landmarks[3], landmarks[4], landmarks[5]],\
           [[chosen_landmark, perpendicular123], [landmarks[5], perpendicular6]],\
           [[landmarks[4], perpendicular6]],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR]

def kpts_acrominal_index():
    return [0, 1, 2, 3, 4, 5]

#---------------------------------------------------------------------

def acromiohumeral_interval(landmarks):
    
    # Define set of all x-coordinates
    acromion_x = [landmarks[5][0], landmarks[6][0], landmarks[7][0], landmarks[8][0], landmarks[9][0], landmarks[10][0], landmarks[11][0]]
    humeral_head_x = [landmarks[12][0], landmarks[13][0], landmarks[14][0], landmarks[15][0], landmarks[16][0], landmarks[17][0], landmarks[18][0], landmarks[19][0]]

    # Determine distance between both sets
    dists = cdist(np.array(acromion_x).reshape(-1, 1), np.array(humeral_head_x).reshape(-1, 1))
    min_dist = np.min(dists)

    return min_dist


def vis_acromiohumeral_interval(landmarks):

    # Define set of all x-coordinates
    acromion_x = [landmarks[5][0], landmarks[6][0], landmarks[7][0], landmarks[8][0], landmarks[9][0], landmarks[10][0], landmarks[11][0]]
    acromion_y = [landmarks[5][1], landmarks[6][1], landmarks[7][1], landmarks[8][1], landmarks[9][1], landmarks[10][1], landmarks[11][1]]
    humeral_head_x = [landmarks[12][0], landmarks[13][0], landmarks[14][0], landmarks[15][0], landmarks[16][0], landmarks[17][0], landmarks[18][0], landmarks[19][0]]
    humeral_head_y = [landmarks[12][1], landmarks[13][1], landmarks[14][1], landmarks[15][1], landmarks[16][1], landmarks[17][1], landmarks[18][1], landmarks[19][1]]

    # Determine distance between both sets
    dists = cdist(np.array(acromion_x).reshape(-1, 1), np.array(humeral_head_x).reshape(-1, 1))

    # Determine landmarks for the minimal distance
    min_acromion_idx, min_humeral_idx = np.unravel_index(np.argmin(dists), dists.shape)	

    return [landmarks[5], landmarks[6], landmarks[7], landmarks[8], landmarks[9], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14], landmarks[15], landmarks[16], landmarks[17], landmarks[18], landmarks[19]],\
           [[[acromion_x[min_acromion_idx], acromion_y[min_acromion_idx]], [humeral_head_x[min_humeral_idx], acromion_y[min_acromion_idx]]]],\
           [[[humeral_head_x[min_humeral_idx], acromion_y[min_acromion_idx]], [humeral_head_x[min_humeral_idx], humeral_head_y[min_humeral_idx]]]],\
           [POINT_COLOR,]*15,\
           [LINE_COLOR],\
           [POINT_COLOR]


def kpts_acromiohumeral_interval():
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


# %% SHOULDER Y MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_shoulder_y = [
    ['1', 'Lateral edge of the acromion lower surface'],
    ['2', 'Medial edge of the acromion lower surface'],
    ['3', 'Midpoint of the acromion lower border (highest point)'],
    ['4', 'Equidistant point 1 along the acromion undersurface between landmarks 1 and 3'],
    ['5', 'Equidistant point 2 along the acromion undersurface between landmarks 1 and 3'],
    ['6', 'Equidistant point 3 along the acromion undersurface between landmarks 1 and 3'],
    ['7', 'Equidistant point 1 along the acromion undersurface between landmarks 2 and 3'],
    ['8', 'Equidistant point 2 along the acromion undersurface between landmarks 2 and 3'],
    ['9', 'Equidistant point 3 along the acromion undersurface between landmarks 2 and 3'],
    ['10', 'Lower edge of the coracoid process - point 1'],
    ['11', 'Lower edge of the coracoid process - point 2'],
    ['12', 'Lower edge of the coracoid process - point 3'],
    ['13', 'Equidistant point 1 along the upper surface of the humeral head'],
    ['14', 'Equidistant point 2 along the upper surface of the humeral head'],
    ['15', 'Equidistant point 3 along the upper surface of the humeral head'],
    ['16', 'Equidistant point 4 along the upper surface of the humeral head'],
    ['17', 'Equidistant point 5 along the upper surface of the humeral head'],
    ['18', 'Equidistant point 6 along the upper surface of the humeral head'],
    ['19', 'Equidistant point 7 along the upper surface of the humeral head'],
    ['20', 'Equidistant point 8 along the upper surface of the humeral head'],
    ['21', 'Equidistant point 9 along the upper surface of the humeral head'],
    ['22', 'Equidistant point 10 along the upper surface of the humeral head'],
    ['23', 'Equidistant point 11 along the upper surface of the humeral head'],
    ['24', 'Equidistant point 12 along the upper surface of the humeral head'],
    ['25', 'Equidistant point 13 along the upper surface of the humeral head'],
    ['26', 'Equidistant point 14 along the upper surface of the humeral head'],
    ['27', 'Equidistant point 15 along the upper surface of the humeral head']
]

#---------------------------------------------------------------------

def acrominal_slope(landmarks):
    
    # Calculate the axes
    acrominal_line1 = landmarks[0] - landmarks[2]
    acrominal_line2 = landmarks[1] - landmarks[2]

    # Calculate the angle
    angle_deg = calculate_angle(acrominal_line1, acrominal_line2)
    
    return angle_deg


def vis_acrominal_slope(landmarks):

    return [landmarks[0], landmarks[1], landmarks[2]],\
           [[landmarks[0], landmarks[2]], [landmarks[1], landmarks[2]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_acrominal_slope():
    return [0, 1, 2]


#---------------------------------------------------------------------

def acrominal_tilt(landmarks):
    
    # Calculate the axes
    acromion_undersurface = landmarks[0] - landmarks[1]
    acrominal_cor = np.mean([landmarks[9],landmarks[10],landmarks[11]], axis=0) - landmarks[2]

    # Calculate the angle
    angle_deg = calculate_angle(acromion_undersurface, acrominal_cor)
    
    return angle_deg


def vis_acrominal_tilt(landmarks):

    mean_acrominal_cor = np.mean([landmarks[9],landmarks[10],landmarks[11]], axis=0)

    return [landmarks[0], landmarks[1], landmarks[2], landmarks[9], landmarks[10], landmarks[11]],\
           [[landmarks[0], landmarks[1]], [mean_acrominal_cor, landmarks[2]]],\
           [[mean_acrominal_cor, landmarks[9]], [mean_acrominal_cor, landmarks[10]], [mean_acrominal_cor, landmarks[11]]],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_acrominal_tilt():
    return [0, 1, 2, 9, 10, 11]


#---------------------------------------------------------------------

def acromiohumeral_distance(landmarks):
    
    # Define set of landmarks
    acromion_landmarks = [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7], landmarks[8]]
    humeral_head_landmarks = [landmarks[12], landmarks[13], landmarks[14], landmarks[15], landmarks[16], landmarks[17], landmarks[18], landmarks[19], landmarks[20], landmarks[21], landmarks[22], landmarks[23], landmarks[24], landmarks[25], landmarks[26]]

    # Determine distance between both sets
    dists = cdist(np.array(acromion_landmarks), np.array(humeral_head_landmarks))
    min_dist = np.min(dists)

    return min_dist


def vis_acromiohumeral_distance(landmarks):

    # Define set of landmarks
    acromion_landmarks = [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7], landmarks[8]]
    humeral_head_landmarks = [landmarks[12], landmarks[13], landmarks[14], landmarks[15], landmarks[16], landmarks[17], landmarks[18], landmarks[19], landmarks[20], landmarks[21], landmarks[22], landmarks[23], landmarks[24], landmarks[25], landmarks[26]]

    # Determine distance between both sets
    dists = cdist(np.array(acromion_landmarks), np.array(humeral_head_landmarks))

    # Determine landmarks for the minimal distance
    min_acromion_idx, min_humeral_idx = np.unravel_index(np.argmin(dists), dists.shape)	

    print(min_acromion_idx, min_humeral_idx)

    return [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7], landmarks[8], landmarks[12], landmarks[13], landmarks[14], landmarks[15], landmarks[16], landmarks[17], landmarks[18], landmarks[19], landmarks[20], landmarks[21], landmarks[22], landmarks[23], landmarks[24], landmarks[25], landmarks[26]],\
           [[acromion_landmarks[min_acromion_idx], humeral_head_landmarks[min_humeral_idx]]],\
           [],\
           [LINE_COLOR,]*24,\
           [LINE_COLOR],\
           []


def kpts_acromiohumeral_distance():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


# %% KNEE AXIAL MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_knee_axial = [
    ['1', 'Deepest point of the intercondylar sulcus'],
    ['2', 'Equidistant points in articular surface of lateral condyle'],
    ['3', 'Equidistant points in articular surface of lateral condyle'],
    ['4', 'Apex of lateral condyle'],
    ['5', 'Equidistant points in articular surface of medial condyle'],
    ['6', 'Equidistant points in articular surface of medial condyle'],
    ['7', 'Apex of medial condyle'],
    ['8', 'Apex of the articular patellar ridge'],
    ['9', 'Equidistant points in articular surface of the patella’s lateral articular facet'],
    ['10', 'Equidistant points in articular surface of the patella’s lateral articular facet'],
    ['11', 'Equidistant points in articular surface of the patella’s lateral articular facet'],
    ['12', 'Lateral edge of the patella'],
    ['13', 'Equidistant points in articular surface of the patella’s medial articular facet'],
    ['14', 'Equidistant points in articular surface of the patella’s medial articular facet'],
    ['15', 'Medial edge of the patella'],
    ['16', 'Equidistant points along anterior patellar surface '],
    ['17', 'Equidistant points along anterior patellar surface '],
    ['18', 'Equidistant points along anterior patellar surface '],
    ['19', 'Equidistant points along anterior patellar surface ']
]

#---------------------------------------------------------------------

def sulcus_angle(landmarks):
    
    # Calculate the axes
    lateral_condyle = landmarks[0] - landmarks[3]
    medial_condyle = landmarks[0] - landmarks[6]

    # Calculate the angle
    angle_deg = calculate_angle(lateral_condyle, medial_condyle)
    
    return angle_deg


def vis_sulcus_angle(landmarks):
    
    return [landmarks[0], landmarks[3], landmarks[6]],\
           [[landmarks[0], landmarks[3]], [landmarks[0], landmarks[6]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_sulcus_angle():
    return [0, 3, 6]

#---------------------------------------------------------------------

def patella_tilting_angle(landmarks):
    
    # Calculate the axes
    anterior_interconylar_line = landmarks[3] - landmarks[6]
    patellar_transverse_axis = landmarks[11] - landmarks[14]

    # Calculate the angle
    angle_deg = calculate_angle(anterior_interconylar_line, patellar_transverse_axis)
    
    return angle_deg


def vis_patella_tilting_angle(landmarks):
    
    return [landmarks[3], landmarks[6], landmarks[11], landmarks[14]],\
           [[landmarks[3], landmarks[6]], [landmarks[11], landmarks[14]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_patella_tilting_angle():
    return [3, 6, 11, 14]

#---------------------------------------------------------------------

def lateral_patellofemoral_angle(landmarks):
    
    # Calculate the axes
    anterior_interconylar_line = landmarks[3] - landmarks[6]
    patellar_lateral_facet = landmarks[11] - landmarks[7]

    # Calculate the angle
    angle_deg = calculate_angle(anterior_interconylar_line, patellar_lateral_facet)
    
    return angle_deg


def vis_lateral_patellofemoral_angle(landmarks):
    
    return [landmarks[3], landmarks[6], landmarks[7], landmarks[11]],\
           [[landmarks[3], landmarks[6]], [landmarks[7], landmarks[11]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_lateral_patellofemoral_angle():
    return [3, 6, 7, 11]

#---------------------------------------------------------------------

def congruence_angle(landmarks):
    
    # Calculate the axes
    lateral_condyle = landmarks[0] - landmarks[3]
    medial_condyle = landmarks[0] - landmarks[6]
    center_line = landmarks[0] - landmarks[7]

    # Calculate the angle
    suculus_angle_deg = calculate_angle(lateral_condyle, medial_condyle)
    center_angle_deg_1 = calculate_angle(lateral_condyle, center_line)
    center_angle_deg_2 = calculate_angle(medial_condyle, center_line)

    congruence_angle_deg = suculus_angle_deg/2 - np.minimum(center_angle_deg_1, center_angle_deg_2)

    return congruence_angle_deg


def vis_congruence_angle(landmarks):
    
    return [landmarks[0], landmarks[3], landmarks[6], landmarks[7]],\
           [[landmarks[0], landmarks[3]], [landmarks[0], landmarks[6]], [landmarks[0], landmarks[7]]],\
           [],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           []

def kpts_congruence_angle():
    return [0, 3, 6, 7]

#---------------------------------------------------------------------


# %% KNEE AP MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_knee_ap = [
    ['1', 'Tibia, lateral plateau (lateral edge)'],
    ['2', 'Tibia, lateral plateau (equidistant along anterior edge)'],
    ['3', 'Tibia, lateral plateau (equidistant along anterior edge)'],
    ['4', 'Tibia, lateral plateau (equidistant along anterior edge)'],
    ['5', 'Tibia, lateral plateau (before intercondylar eminence)'],
    ['6', 'Tibia, medial plateau (after intercondylar eminence)'],
    ['7', 'Tibia, medial plateau (equidistant along anterior edge)'],
    ['8', 'Tibia, medial plateau (equidistant along anterior edge)'],
    ['9', 'Tibia, medial plateau (equidistant along anterior edge)'],
    ['10', 'Tibia, medial plateau (medial edge)'],
    ['11', 'Femur, lateral condyle'],
    ['12', 'Femur, lateral condyle'],
    ['13', 'Femur, lateral condyle'],
    ['14', 'Femur, medial condyle'],
    ['15', 'Femur, medial condyle'],
    ['16', 'Femur, medial condyle']
]

#---------------------------------------------------------------------

def medial_joint_space_width(landmarks):

    # Set up both landmark sets
    medial_condile_set = np.array([landmarks[13], landmarks[14], landmarks[15]])
    medial_plateau_set = np.array([landmarks[6], landmarks[7], landmarks[8]])

    # Determine closest points
    dists = cdist(medial_condile_set, medial_plateau_set)
    min_plateau_dist = np.argmin(dists, axis=0)
    min_condile_dist = np.argmin(dists, axis=1)

    # Construct lines
    condile_lines = [[medial_condile_set[i], np.array([medial_plateau_set[min_condile_dist[i]][0], medial_condile_set[i][1]])] for i in range(len(medial_condile_set))]
    plateau_lines = [[medial_plateau_set[i], np.array([medial_condile_set[min_plateau_dist[i]][0], medial_plateau_set[i][1]])] for i in range(len(medial_plateau_set))]

    # Calculate the distances for both condile and plateau lines
    distance = np.mean([np.linalg.norm(condile_lines[i][0] - condile_lines[i][1]) for i in range(len(condile_lines))] +\
                         [np.linalg.norm(plateau_lines[i][0] - plateau_lines[i][1]) for i in range(len(plateau_lines))])

    return distance


def vis_medial_joint_space_width(landmarks):

    # Set up both landmark sets
    medial_condile_set = [landmarks[13], landmarks[14], landmarks[15]]
    medial_plateau_set = [landmarks[6], landmarks[7], landmarks[8]]

    # Determine closest points
    dists = cdist(np.array(medial_condile_set), np.array(medial_plateau_set))
    min_condile_dist = np.argmin(dists, axis=1)
    min_plateau_dist = np.argmin(dists, axis=0)

    # Construct lines
    condile_lines = [(medial_condile_set[i], [medial_plateau_set[min_condile_dist[i]][0], medial_condile_set[i][1]]) for i in range(len(medial_condile_set))]
    plateau_lines = [(medial_plateau_set[i], [medial_condile_set[min_plateau_dist[i]][0], medial_plateau_set[i][1]]) for i in range(len(medial_plateau_set))]


    return medial_condile_set + medial_plateau_set,\
           condile_lines + plateau_lines,\
           [],\
           [LINE_COLOR,]*len(medial_condile_set) + [LINE_COLOR]*len(medial_plateau_set),\
           [LINE_COLOR,]*len(condile_lines) + [LINE_COLOR]*len(plateau_lines),\
           []

def kpts_medial_joint_space_width():
    return [6, 7, 8, 13, 14, 15]

#---------------------------------------------------------------------

def lateral_joint_space_width(landmarks):

    # Set up both landmark sets
    lateral_condile_set = np.array([landmarks[10], landmarks[11], landmarks[12]])
    lateral_plateau_set = np.array([landmarks[2], landmarks[3], landmarks[4]])

    # Determine closest points
    dists = cdist(lateral_condile_set, lateral_plateau_set)
    min_condile_dist = np.argmin(dists, axis=1)
    min_plateau_dist = np.argmin(dists, axis=0)

    # Construct lines
    condile_lines = [(lateral_condile_set[i], [lateral_plateau_set[min_condile_dist[i]][0], lateral_condile_set[i][1]]) for i in range(len(lateral_condile_set))]
    plateau_lines = [(lateral_plateau_set[i], [lateral_condile_set[min_plateau_dist[i]][0], lateral_plateau_set[i][1]]) for i in range(len(lateral_plateau_set))]

    # Calculate the distances for both condile and plateau lines
    distance = np.mean([np.linalg.norm(condile_lines[i][0] - condile_lines[i][1]) for i in range(len(condile_lines))] +\
                         [np.linalg.norm(plateau_lines[i][0] - plateau_lines[i][1]) for i in range(len(plateau_lines))])

    
    return distance


def vis_lateral_joint_space_width(landmarks):

    # Set up both landmark sets
    lateral_condile_set = [landmarks[10], landmarks[11], landmarks[12]]
    lateral_plateau_set = [landmarks[2], landmarks[3], landmarks[4]]

    # Determine closest points
    dists = cdist(np.array(lateral_condile_set), np.array(lateral_plateau_set))
    min_condile_dist = np.argmin(dists, axis=1)
    min_plateau_dist = np.argmin(dists, axis=0)

    # Construct lines
    condile_lines = [(lateral_condile_set[i], [lateral_plateau_set[min_condile_dist[i]][0], lateral_condile_set[i][1]]) for i in range(len(lateral_condile_set))]
    plateau_lines = [(lateral_plateau_set[i], [lateral_condile_set[min_plateau_dist[i]][0], lateral_plateau_set[i][1]]) for i in range(len(lateral_plateau_set))]


    return lateral_condile_set + lateral_plateau_set,\
           condile_lines + plateau_lines,\
           [],\
           [LINE_COLOR,]*len(lateral_condile_set) + [LINE_COLOR]*len(lateral_plateau_set),\
           [LINE_COLOR,]*len(condile_lines) + [LINE_COLOR]*len(plateau_lines),\
           []

def kpts_lateral_joint_space_width():
    return [2, 3, 4, 10, 11, 12]

#---------------------------------------------------------------------


# %% KNEE LATERAL MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_knee_lateral = [
    ['1', 'Patella, base (upper border)'],
    ['2', 'Patella, lower border of articular surface'],
    ['3', 'Patella, apex (lower border)'],
    ['4', 'Tibia, tibial tuverosity'],
    ['5', 'Tibia, posterior cortex'],
    ['6', 'Tibia, posterior cortex'],
    ['7', 'Tibia, anterior edge of tibial plateau'],
    ['8', 'Tibia, posterior edge of tibial plateau'],
    ['9', 'Tibia, center'],
    ['10', 'Tibia, center'],
    ['11', 'Tibia, posterior edge of tibial plateau'],
    ['12', 'Tibia, posterior edge of tibial plateau'],
    ['13', 'Tibia, posterior edge of tibial plateau'],
    ['14', 'Tibia, posterior edge of tibial plateau'],
    ['15', 'Tibia, posterior edge of tibial plateau'],
    ['16', 'Tibia, posterior cortex'],
    ['17', 'Tibia, posterior cortex'],
    ['18', 'Tibia, center'],
    ['19', 'Tibia, center'],
    ['20', 'Tibia, superior cortex'],
    ['21', 'Tibia, superior cortex'],
    ['22', 'Tibia, superior cortex'],
    ['23', 'Tibia, superior cortex'],
    ['24', 'Patella, base (upper border)'],
    ['25', 'Tibia, anterior edge of tibial plateau']
]

#---------------------------------------------------------------------

def insall_salvati_ratio(landmarks):
    
    patellar_tendon_line = landmarks[2] - landmarks[3]
    patellar_line = landmarks[23] - landmarks[2]

    # Calculate the measurement
    patellar_tendon_length = np.linalg.norm(patellar_tendon_line)
    patellar_length = np.linalg.norm(patellar_line)

    # Calculate the ratio
    ratio = patellar_tendon_length/patellar_length    
    
    return ratio


def vis_insall_salvati_ratio(landmarks):
    
    return [landmarks[23], landmarks[2], landmarks[3]],\
           [[landmarks[23], landmarks[2]], [landmarks[2], landmarks[3]]],\
           [],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_insall_salvati_ratio():
    return [23, 2, 3]

#---------------------------------------------------------------------

def modified_insall_salvati_ratio(landmarks):
    
    patellar_tendon_line = landmarks[1] - landmarks[3]
    patellar_line = landmarks[0] - landmarks[1]

    # Calculate the measurement
    patellar_tendon_length = np.linalg.norm(patellar_tendon_line)
    patellar_length = np.linalg.norm(patellar_line)

    # Calculate the ratio
    ratio = patellar_tendon_length/patellar_length    
    
    return ratio


def vis_modified_insall_salvati_ratio(landmarks):
    
    return [landmarks[0], landmarks[1], landmarks[3]],\
           [[landmarks[0], landmarks[1]], [landmarks[1], landmarks[3]]],\
           [],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_modified_insall_salvati_ratio():
    return [0, 1, 3]

#---------------------------------------------------------------------

def caton_deschamps_index(landmarks):
    
    patella_to_tibial_plateau_length = np.linalg.norm(landmarks[24] - landmarks[1])
    patellar_articular_surface_length = np.linalg.norm(landmarks[0] - landmarks[1])

    return patella_to_tibial_plateau_length/patellar_articular_surface_length


def vis_caton_deschamps_index(landmarks):
    
    return [landmarks[24], landmarks[0], landmarks[1]],\
           [[landmarks[24], landmarks[1]], [landmarks[0], landmarks[1]]],\
           [],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_caton_deschamps_index():
    return [0, 1, 24]

#---------------------------------------------------------------------

def blackburn_peel_ratio(landmarks):
    
    patella_to_tibial_plateau_length = np.linalg.norm(landmarks[24] - landmarks[1])
    articular_surface1, articular_surface2 = best_fit_line_from_landmarks([landmarks[6], landmarks[7], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14]])
    patella_articular_surface_height = perpendicular_point_on_line(articular_surface1, articular_surface2, landmarks[1])
    patellar_articular_surface_length = np.linalg.norm(patella_articular_surface_height - landmarks[1])

    return patella_to_tibial_plateau_length/patellar_articular_surface_length


def vis_blackburn_peel_ratio(landmarks):
    
    articular_surface1, articular_surface2 = best_fit_line_from_landmarks([landmarks[6], landmarks[7], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14]])
    patella_articular_surface_height = perpendicular_point_on_line(articular_surface1, articular_surface2, landmarks[1])

    return [landmarks[6], landmarks[7], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14], landmarks[24], landmarks[0], landmarks[1]],\
           [[landmarks[0], landmarks[1]], [patella_articular_surface_height, landmarks[1]]],\
           [[articular_surface1, articular_surface2]],\
           [POINT_COLOR,]*10,\
           [LINE_COLOR, LINE_COLOR],\
           [POINT_COLOR]

def kpts_blackburn_peel_ratio():
    return [0, 1, 6, 7, 10, 11, 12, 13, 14, 24]

#---------------------------------------------------------------------

def patella_morphology_ratio(landmarks):
    
    patella_length = np.linalg.norm(landmarks[2] - landmarks[0])
    patellar_articular_surface_length = np.linalg.norm(landmarks[0] - landmarks[1])

    return patella_length/patellar_articular_surface_length


def vis_patella_morphology_ratio(landmarks):
    
    return [landmarks[2], landmarks[0], landmarks[1]],\
           [[landmarks[2], landmarks[0]], [landmarks[0], landmarks[1]]],\
           [],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_patella_morphology_ratio():
    return [0, 1, 2]

#---------------------------------------------------------------------

def posterior_posterior_tibial_slope(landmarks):
    
    posterior_tibial_cortex = landmarks[4] - landmarks[5]
    tibial_plateau = landmarks[7] - landmarks[6]

    # Calculate the angle
    angle_deg = calculate_angle(posterior_tibial_cortex, tibial_plateau)
    angle_deg = np.minimum(angle_deg, 180-angle_deg)
    
    return 90-angle_deg


def vis_posterior_posterior_tibial_slope(landmarks):
    
    return [landmarks[4], landmarks[5], landmarks[6], landmarks[7]],\
           [[landmarks[4], landmarks[5]], [landmarks[6], landmarks[7]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_posterior_posterior_tibial_slope():
    return [4, 5, 6, 7]

#---------------------------------------------------------------------

def medial_posterior_tibial_slope(landmarks):
    
    posterior_tibial_cortex = landmarks[8] - landmarks[9]
    tibial_plateau = landmarks[7] - landmarks[6]

    # Calculate the angle
    angle_deg = calculate_angle(posterior_tibial_cortex, tibial_plateau)
    angle_deg = np.minimum(angle_deg, 180-angle_deg)
    
    return 90-angle_deg


def vis_medial_posterior_tibial_slope(landmarks):
    
    return [landmarks[8], landmarks[9], landmarks[6], landmarks[7]],\
           [[landmarks[8], landmarks[9]], [landmarks[6], landmarks[7]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_medial_posterior_tibial_slope():
    return [8, 9, 6, 7]

#---------------------------------------------------------------------

def anterior_posterior_tibial_slope(landmarks):
    
    posterior_tibial_cortex = landmarks[19] - landmarks[22]
    tibial_plateau = landmarks[7] - landmarks[6]

    # Calculate the angle
    angle_deg = calculate_angle(posterior_tibial_cortex, tibial_plateau)
    angle_deg = np.minimum(angle_deg, 180-angle_deg)
    
    return 90-angle_deg


def vis_anterior_posterior_tibial_slope(landmarks):
    
    return [landmarks[19], landmarks[22], landmarks[6], landmarks[7]],\
           [[landmarks[19], landmarks[22]], [landmarks[6], landmarks[7]]],\
           [],\
           [LINE_COLOR, LINE_COLOR, LINE_COLOR, LINE_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_anterior_posterior_tibial_slope():
    return [19, 22, 6, 7]

#---------------------------------------------------------------------

# %% HIP AP MEASUREMENTS
#---------------------------------------------------------------------

kpt_names_hip_ap = [
    ['1', '1 right'],
    ['2', '2 right'],
    ['3', '3 right'],
    ['4', '4 right'],    
    ['5', '5 left'], 
    ['6', '6 left'], 
    ['7', '7 left'], 
    ['8', '8 left'],
]

#---------------------------------------------------------------------

def ac_angle_right(landmarks):
    
    hilgenreiner_line = landmarks[3] - landmarks[4]
    hip_line = landmarks[2] - landmarks[3]

    # Calculate the angle
    angle_deg = calculate_angle(hilgenreiner_line, hip_line)  
    angle_deg = np.minimum(angle_deg, 180-angle_deg)  
    
    return angle_deg


def vis_ac_angle_right(landmarks):
    
    return [landmarks[2], landmarks[3], landmarks[4]],\
           [[landmarks[2], landmarks[3]], [landmarks[3], landmarks[4]]],\
           [],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_ac_angle_right():
    return [2, 3, 4]

#---------------------------------------------------------------------

def ac_angle_left(landmarks):
    
    hilgenreiner_line = landmarks[3] - landmarks[4]
    hip_line = landmarks[4] - landmarks[5]

    # Calculate the angle
    angle_deg = calculate_angle(hilgenreiner_line, hip_line)  
    angle_deg = np.minimum(angle_deg, 180-angle_deg)  
    
    return angle_deg


def vis_ac_angle_left(landmarks):
    
    return [landmarks[3], landmarks[4], landmarks[5]],\
           [[landmarks[3], landmarks[4]], [landmarks[4], landmarks[5]]],\
           [],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           []

def kpts_ac_angle_left():
    return [3, 4, 5]

#---------------------------------------------------------------------

def reimers_index_right(landmarks):
    
    # Don't calculate if the hip extends beyond the humeral head
    if landmarks[0][1] > landmarks[2][1]: return 0

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perkins_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[2])
    A = np.abs(perkins_point[1] - humeral_head_lateral_point[1])
    humeral_head_length = np.abs(humeral_head_medial_point[1] - humeral_head_lateral_point[1])

    # Calculate the ratio
    ratio = A/humeral_head_length*100
    
    return ratio


def vis_reimers_index_right(landmarks):

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perkins_point = perpendicular_point_on_line(landmarks[2], landmarks[5], landmarks[2])

    humeral_head_mean = np.mean([landmarks[0][0], landmarks[1][0]])
    hip_humeral_head_mean = np.mean([landmarks[0][0], landmarks[2][0]])
    lowest_point = np.max([landmarks[0][0], landmarks[1][0]])
    
    return [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]],\
           [[(humeral_head_mean, humeral_head_medial_point[1]), (humeral_head_mean, humeral_head_lateral_point[1])],\
            [(hip_humeral_head_mean, humeral_head_lateral_point[1]), (hip_humeral_head_mean, perkins_point[1])]],\
           [[(lowest_point, landmarks[0][1]), (landmarks[2][0], humeral_head_lateral_point[1])],\
            [(lowest_point, landmarks[1][1]), (landmarks[2][0], humeral_head_medial_point[1])],\
            [(lowest_point, landmarks[2][1]), (landmarks[2][0], perkins_point[1])],\
            [landmarks[3], landmarks[4]]],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_reimers_index_right():
    return [0, 1, 2, 3, 4]

#---------------------------------------------------------------------

def reimers_index_left(landmarks):
    
    # Don't calculate if the hip extends beyond the humeral head
    if landmarks[7][1] < landmarks[5][1]: return 0

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[7])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[6])
    perkins_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[5])
    A = np.abs(perkins_point[1] - humeral_head_lateral_point[1])
    humeral_head_length = np.abs(humeral_head_medial_point[1] - humeral_head_lateral_point[1])

    # Calculate the ratio
    ratio = A/humeral_head_length*100 
    
    return ratio


def vis_reimers_index_left(landmarks):

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[7])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[6])
    perkins_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[5])

    humeral_head_mean = np.mean([landmarks[6][0], landmarks[7][0]])
    hip_humeral_head_mean = np.mean([landmarks[5][0], landmarks[7][0]])
    lowest_point = np.max([landmarks[6][0], landmarks[7][0]])
    
    return [landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7]],\
           [[(humeral_head_mean, humeral_head_medial_point[1]), (humeral_head_mean, humeral_head_lateral_point[1])],\
            [(hip_humeral_head_mean, humeral_head_lateral_point[1]), (hip_humeral_head_mean, perkins_point[1])]],\
           [[(lowest_point, landmarks[7][1]), (landmarks[5][0], humeral_head_lateral_point[1])],\
            [(lowest_point, landmarks[6][1]), (landmarks[5][0], humeral_head_medial_point[1])],\
            [(lowest_point, landmarks[5][1]), (landmarks[5][0], perkins_point[1])],\
            [landmarks[3], landmarks[4]]],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_reimers_index_left():
    return [3, 4, 5, 6, 7]

#---------------------------------------------------------------------

def humeral_head_length_right(landmarks):
    
    # Don't calculate if the hip extends beyond the humeral head
    if landmarks[0][1] > landmarks[2][1]: return 0

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perkins_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[2])
    A = np.abs(perkins_point[1] - humeral_head_lateral_point[1])
    humeral_head_length = np.abs(humeral_head_medial_point[1] - humeral_head_lateral_point[1])

    # Calculate the ratio
    ratio = A/humeral_head_length*100
    
    return humeral_head_length


def vis_humeral_head_length_right(landmarks):

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perkins_point = perpendicular_point_on_line(landmarks[2], landmarks[5], landmarks[2])

    humeral_head_mean = np.mean([landmarks[0][0], landmarks[1][0]])
    hip_humeral_head_mean = np.mean([landmarks[0][0], landmarks[2][0]])
    lowest_point = np.max([landmarks[0][0], landmarks[1][0]])
    
    return [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]],\
           [[(humeral_head_mean, humeral_head_medial_point[1]), (humeral_head_mean, humeral_head_lateral_point[1])],\
            [(hip_humeral_head_mean, humeral_head_lateral_point[1]), (hip_humeral_head_mean, perkins_point[1])]],\
           [[(lowest_point, landmarks[0][1]), (landmarks[2][0], humeral_head_lateral_point[1])],\
            [(lowest_point, landmarks[1][1]), (landmarks[2][0], humeral_head_medial_point[1])],\
            [(lowest_point, landmarks[2][1]), (landmarks[2][0], perkins_point[1])],\
            [landmarks[3], landmarks[4]]],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_humeral_head_length_right():
    return [0, 1, 2, 3, 4]

#---------------------------------------------------------------------

def A_right(landmarks):
    
    # Don't calculate if the hip extends beyond the humeral head
    if landmarks[0][1] > landmarks[2][1]: return 0

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perkins_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[2])
    A = np.abs(perkins_point[1] - humeral_head_lateral_point[1])
    humeral_head_length = np.abs(humeral_head_medial_point[1] - humeral_head_lateral_point[1])

    # Calculate the ratio
    ratio = A/humeral_head_length*100
    
    return A


def vis_A_right(landmarks):

    humeral_head_lateral_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[0])
    humeral_head_medial_point = perpendicular_point_on_line(landmarks[3], landmarks[4], landmarks[1])
    perkins_point = perpendicular_point_on_line(landmarks[2], landmarks[5], landmarks[2])

    humeral_head_mean = np.mean([landmarks[0][0], landmarks[1][0]])
    hip_humeral_head_mean = np.mean([landmarks[0][0], landmarks[2][0]])
    lowest_point = np.max([landmarks[0][0], landmarks[1][0]])
    
    return [landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]],\
           [[(humeral_head_mean, humeral_head_medial_point[1]), (humeral_head_mean, humeral_head_lateral_point[1])],\
            [(hip_humeral_head_mean, humeral_head_lateral_point[1]), (hip_humeral_head_mean, perkins_point[1])]],\
           [[(lowest_point, landmarks[0][1]), (landmarks[2][0], humeral_head_lateral_point[1])],\
            [(lowest_point, landmarks[1][1]), (landmarks[2][0], humeral_head_medial_point[1])],\
            [(lowest_point, landmarks[2][1]), (landmarks[2][0], perkins_point[1])],\
            [landmarks[3], landmarks[4]]],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR],\
           [LINE_COLOR, LINE_COLOR],\
           [POINT_COLOR, POINT_COLOR, POINT_COLOR, POINT_COLOR]

def kpts_A_right():
    return [0, 1, 2, 3, 4]

#---------------------------------------------------------------------