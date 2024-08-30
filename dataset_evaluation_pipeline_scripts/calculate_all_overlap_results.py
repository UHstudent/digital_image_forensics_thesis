import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

#helpfunction to find the threshold at a target FPR
def find_threshold_at_fpr(fpr, thresholds, target_fpr):
    return np.interp(target_fpr, fpr, thresholds)

#helpfunction to find the highest TPR at FPR target
def find_highest_tpr_at_fpr(fpr, tpr, target_fpr):
    indices = np.where(fpr == target_fpr)[0]
    if indices.size > 0:
        return max(tpr[indices])
    else:
        return np.interp(target_fpr, fpr, tpr)
    
#helpfunction to get true positives, false positives & respective image names
def get_tp_fp(y_true, y_scores, threshold, image_names):
    y_pred = np.array(y_scores) >= threshold
    tp_indices = [i for i, (pred, true) in enumerate(zip(y_pred, y_true)) if pred and true]
    fp_indices = [i for i, (pred, true) in enumerate(zip(y_pred, y_true)) if pred and not true]
    tp_images = [image_names[i] for i in tp_indices]
    fp_images = [image_names[i] for i in fp_indices]
    return tp_indices, fp_indices, tp_images, fp_images

#helpfunction to save data resctructured
def save_data_structured(data, algo, result_path, filename):
    with open(str(result_path) + 'fixed_' + str(filename), 'w') as file:
        file.write('[')
        for i, entry in enumerate(data):
            json.dump(entry, file)
            #only write new line if not last element
            if i < len(data) - 1:
                file.write(',\n')

        file.write(']')

    #sort by k stat
    if algo == "ghostmaps":
        sorted_by_kstat = sorted(data, key=lambda x: x["largest k stat diff"])
    else:
        sorted_by_kstat = sorted(data, key=lambda x: x["k stat diff"])

    #save sorted by k stat to new file
    with open(str(result_path) + 'sorted_kstat_' + str(filename), 'w') as file:
        file.write('[')
        for i, entry in enumerate(sorted_by_kstat):
            json.dump(entry, file)
            #only write new line if not last element
            if i < len(data) - 1:
                file.write(',\n')
        file.write(']')

    #sort by absolute median difference
    sorted_by_abs_median = sorted(data, key=lambda x: x['absolute median difference'])

    #save sorted by k stat to new file
    with open(str(result_path) + 'sorted_abs_med_' + str(filename), 'w') as file:
        file.write('[')
        for i, entry in enumerate(sorted_by_abs_median):
            json.dump(entry, file)
            #only write new line if not last element
            if i < len(data) - 1:
                file.write(',\n')
        file.write(']')


SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#DFAD 2023
#"results_DFAD2023_CocoGlide.json"
#"results_DFAD2023_Columbia.json"
#"results_DFAD2023_Coverage.json"
#"results_DFAD2023_Coverage2.json"
#"results_DFAD2023_IFS.json"
#"results_DFAD2023_IMD2020.json"
#"results_DFAD2023_in_the_wild.json"

coverage = ["results_ghostmaps_Coverage.json","results_noise_Coverage.json","results_probability_3x3_Coverage.json"]
coverage2 = ["results_ghostmaps_Coverage2.json","results_noise_Coverage2.json","results_probability_3x3_Coverage2.json"]
cocoglide = ["results_ghostmaps_CocoGlide.json","results_noise_CocoGlide.json","results_probability_3x3_CocoGlide.json"]
ifs = ["results_ghostmaps_IFS.json","results_noise_IFS.json","results_probability_3x3_IFS.json"]
columbia = ["results_ghostmaps_Columbia.json","results_noise_Columbia.json","results_probability_3x3_Columbia.json"]
imd2020 = ["results_ghostmaps_IMD2020.json","results_noise_IMD2020.json","results_probability_3x3_IMD2020.json"]
wild = ["results_ghostmaps_in_the_wild.json","results_noise_in_the_wild.json","results_probability_3x3_in_the_wild.json"]

files_mmfusion = ["results_MM-fusion_IMD2020.json","results_MM-fusion_IFS.json","results_MM-fusion_Coverage2.json",
                   "results_MM-fusion_Coverage.json","results_MM-fusion_Columbia.json","results_MM-fusion_CocoGlide.json"]

datasets_orig_manip = [coverage,coverage2,cocoglide,ifs,columbia,imd2020]

#calculate overlap/distinctons per traditional technique for k-s stat and abs median diff
for dataset in datasets_orig_manip:
    print(dataset)
    for filename in dataset:
        print(filename)
        parts = filename.split("_")
        last = len(parts)-1
        dataset = parts[last][:-5]
        algo = parts[1]
        result_path = "results/" + algo + "/" + dataset

        if algo == "noise":
            algo_name = "Noise Wavelets"
            
        elif algo == "probability":
            algo_name = "Probability maps"
        else:
            algo_name = "Ghostmaps"

        #add slash so filename can be easily added
        result_path = result_path + "/"

        #load the JSON data from a file
        with open(filename, 'r') as file:
            raw_data = file.read()

        #fix the formatting by wrapping the objects in an array
        raw_data = '[\n' + raw_data.replace('}\n{', '},\n{') + ']'

        #parse the fixed JSON data
        data = json.loads(raw_data)

        #save data in propper json formats
        save_data_structured(data, algo, result_path, filename)


        #extract the necessary values
        y_true = [1 if d["type"] == "manipulated" else 0 for d in data]
        abs_median_diff = [d["absolute median difference"] for d in data]
        
        if algo == "ghostmaps":
            largest_k_stat_diff = [d["largest k stat diff"] for d in data]
            image_names = [d["image name"] for d in data]
        else:
            largest_k_stat_diff = [d["k stat diff"] for d in data]
            image_names = [d["image_name"] for d in data]

        #----------------------------------------------------------------------------------------------------------
        #calculate ROC, FPR & TPR for results and calculate needed statistics -------------------------------------

        #calculate ROC for absolute median difference
        fpr_abs, tpr_abs, thresholds_abs = roc_curve(y_true, abs_median_diff)
        roc_auc_abs = auc(fpr_abs, tpr_abs)

        #calculate ROC for largest k stat diff
        fpr_kstat, tpr_kstat, thresholds_kstat = roc_curve(y_true, largest_k_stat_diff)
        roc_auc_kstat = auc(fpr_kstat, tpr_kstat)

        #calculate and save threshold values per FPR -------------------------------------------------------------------

        thresholdsssss = np.linspace(1.00, 0.00, 21)

        thresholds_abs_highlight = [find_threshold_at_fpr(fpr_abs, thresholds_abs, fpr) for fpr in thresholdsssss]
        thresholds_kstat_highlight = [find_threshold_at_fpr(fpr_kstat, thresholds_kstat, fpr) for fpr in thresholdsssss]

        #save threshold values per FPR
        with open(str(result_path) + 'threshold_per_FPR_incresase.json', 'w') as file:
            file.write('[')
            for i, fpr in enumerate(thresholdsssss):
                json.dump((f'FPR: {fpr:.2f}, Median Difference Threshold: {thresholds_abs_highlight[i]:.5f}, K-S Stat Threshold: {thresholds_kstat_highlight[i]:.5f}'), file)
                #only write new line if not last element
                if i < len(data) - 1:
                    file.write(',\n')
            file.write(']')

        #-------------------------------------------------------------------------------------------------------
        #calculate overlap for med diff and k-s stat -----------------------------------------------------------


        highlight_fprs = [0, 0.05, 0.1, 0.2]
        overlap_results = {}

        for fpr_target in highlight_fprs:
            threshold_abs = find_threshold_at_fpr(fpr_abs, thresholds_abs, fpr_target)
            threshold_kstat = find_threshold_at_fpr(fpr_kstat, thresholds_kstat, fpr_target)

            tp_indices_abs, fp_indices_abs, tp_images_abs, fp_images_abs = get_tp_fp(y_true, abs_median_diff, threshold_abs, image_names)
            tp_indices_kstat, fp_indices_kstat, tp_images_kstat, fp_images_kstat = get_tp_fp(y_true, largest_k_stat_diff, threshold_kstat, image_names)

            overlap_tp_indices = list(set(tp_indices_abs).intersection(tp_indices_kstat))
            overlap_tp_images = [image_names[i] for i in overlap_tp_indices]
            
            unique_tp_abs = list(set(tp_indices_abs) - set(tp_indices_kstat))
            unique_tp_abs_images = [image_names[i] for i in unique_tp_abs]
            
            unique_tp_kstat = list(set(tp_indices_kstat) - set(tp_indices_abs))
            unique_tp_kstat_images = [image_names[i] for i in unique_tp_kstat]

            overlap_fp_indices = list(set(fp_indices_abs).intersection(fp_indices_kstat))
            overlap_fp_images = [image_names[i] for i in overlap_fp_indices]
            
            unique_fp_abs = list(set(fp_indices_abs) - set(fp_indices_kstat))
            unique_fp_abs_images = [image_names[i] for i in unique_fp_abs]
            
            unique_fp_kstat = list(set(fp_indices_kstat) - set(fp_indices_abs))
            unique_fp_kstat_images = [image_names[i] for i in unique_fp_kstat]

            images_unique_detections = list(set(tp_images_abs + tp_images_kstat))
            
            total_false_positives_images = list(set(fp_images_abs + fp_images_kstat))

            overlap_results[f"{int(fpr_target * 100)}% FPR"] = {
                "Correctly detected by K-S": len(tp_indices_kstat),
                "Correctly detected by Abs Median": len(tp_indices_abs),
                "Total unique detections": {
                    "Count": len(images_unique_detections),
                    "Images": images_unique_detections
                },
                "Overlap": {
                    "Count": len(overlap_tp_indices),
                    "Images": overlap_tp_images,
                },
                "Unique to K-S": {
                    "Count": len(unique_tp_kstat),
                    "Images": unique_tp_kstat_images,
                },
                "Unique to Abs Median": {
                    "Count": len(unique_tp_abs),
                    "Images": unique_tp_abs_images,
                },
                "False positives by K-S": len(fp_indices_kstat),
                "False positives by Abs Median": len(fp_indices_abs),
                "Total False positives": {
                    "Count": len(total_false_positives_images),
                    "Images": total_false_positives_images,
                },
                "False Positives": {
                    "Overlap": {
                        "Count": len(overlap_fp_indices),
                        "Images": overlap_fp_images,
                    },
                    "Unique to K-S": {
                        "Count": len(unique_fp_kstat),
                        "Images": unique_fp_kstat_images,
                    },
                    "Unique to Abs Median": {
                        "Count": len(unique_fp_abs),
                        "Images": unique_fp_abs_images,
                    },
                }
            }


        #save the overlap results to a JSON file
        with open(result_path + 'overlap_results.json', 'w') as file:
            json.dump(overlap_results, file, indent=4)

        #----------------------------------------------------------------------------------------------
        #plot ROC curves ------------------------------------------------------------------------------
        tpr_abs_highlight = [find_highest_tpr_at_fpr(fpr_abs, tpr_abs, fpr) for fpr in highlight_fprs]
        tpr_kstat_highlight = [find_highest_tpr_at_fpr(fpr_kstat, tpr_kstat, fpr) for fpr in highlight_fprs]

        plt.figure(figsize=(10, 8))
        plt.plot(fpr_abs, tpr_abs, color='blue', lw=2, label='Absolute Median Difference (AUC = %0.2f)' % roc_auc_abs)
        plt.plot(fpr_kstat, tpr_kstat, color='magenta', lw=2, label='K-S Stat Difference (AUC = %0.2f)' % roc_auc_kstat)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

        color_markers = ['green', 'gold', 'darkorange', "red"]

        #highlight TPR for specific FPR values
        for i, fpr in enumerate(highlight_fprs):
            plt.plot(fpr, tpr_abs_highlight[i], 'o', color=color_markers[i],label=f'Median difference ({fpr:.2f}, {tpr_abs_highlight[i]:.2f})')
            plt.plot(fpr, tpr_kstat_highlight[i], 'd', color=color_markers[i], label=f'K-S statistic ({fpr:.2f}, {tpr_kstat_highlight[i]:.2f})')



        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(dataset + ': '+ algo_name + ' ROC curve')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(result_path + "ROC_curve_" + algo + "_" + dataset + ".png")

        #-------------------------------------------------------------------------------------------------

        

        


#calculate overlap/distinctons per traditional technique for k-s stat and abs median diff for wild dataset
#helperfunction estimating AUC for wild dataset
def estimate_auc(fpr, tpr):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    #trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return auc


for filename in wild:
    print(filename)
    parts = filename.split("_")
    last = len(parts)-1
    dataset = parts[last][:-5]
    algo = parts[1]
    result_path = "results/" + algo + "/" + dataset

    if algo == "noise":
        algo_name = "Noise Wavelets"
        
    elif algo == "probability":
        algo_name = "Probability maps"
    else:
        algo_name = "Ghostmaps"

    #add slash so filename can be easily added
    result_path = result_path + "/"

    #load the JSON data from a file
    with open(filename, 'r') as file:
        raw_data = file.read()

    #fix the formatting by wrapping the objects in an array
    raw_data = '[\n' + raw_data.replace('}\n{', '},\n{') + ']'

    #parse the fixed JSON data
    data = json.loads(raw_data)

    #save data in propper json formats
    save_data_structured(data, algo, result_path, filename)

    #extract the necessary values
    y_true = [1 if d["type"] == "manipulated" else 0 for d in data]
    abs_median_diff = [d["absolute median difference"] for d in data]
    abs_median_diff = np.array(abs_median_diff)

    if algo == "ghostmaps":
        largest_k_stat_diff = [d["largest k stat diff"] for d in data]
        image_names = [d["image name"] for d in data]
        #ghost thresholds from IMD2020, from 0% FPR to 100FPR in steps of 0.5, ghostmaps are normalized between 0-1, that is why mediand diff is uncharacteristically low
        x_axis_medt = [0.80369,0.11093,0.06949,0.05891,0.05135,0.04510,0.04008,0.03688,0.03373,0.03005,0.02603,0.02296,0.02140,0.01891,0.01659,0.01416,0.01206,0.01018,0.00861,0.00616,0.00]
        x_axis_kst = [0.52385,0.34948,0.29321,0.25970,0.23638,0.21668,0.20360,0.18584,0.17853,0.16908,0.16333,0.15388,0.14499,0.13652,0.12335,0.11615,0.10379,0.09555,0.08437,0.06910,0.03741]
    elif algo == "noise":
        largest_k_stat_diff = [d["k stat diff"] for d in data]
        image_names = [d["image_name"] for d in data]
        #noise thresholds from IMD2020, from 0% FPR to 100FPR in steps of 0.5
        x_axis_medt = [92.44356,24.54286,18.61200,13.88990,11.05910,9.13584,7.27962,6.22347,5.39337,4.69027,4.05360,3.48710,2.77370,2.42940,1.96315,1.63352,1.38086,0.97291,0.63915,0.26149,0.0]
        x_axis_kst = [0.67765,0.49853,0.42416,0.36524,0.32365,0.29788,0.27844,0.25273,0.23692,0.21872,0.20249,0.19062,0.18002,0.16670,0.15578,0.14208,0.13091,0.11423,0.09340,0.07699,0.03072]
    else:
        largest_k_stat_diff = [d["k stat diff"] for d in data]
        image_names = [d["image_name"] for d in data]
        #probability thresholds from IMD2020, from 0% FPR to 100FPR in steps of 0.5
        x_axis_medt = [227.30831,142.97872,88.37271,67.07439,47.75,33.12,24.87,15.18,10.94,8.04,5.84,3.87,3.07,2.18,1.64,1.03,0.58,0.31,0.13,0.04,0.00]
        x_axis_kst = [0.52877,0.32723,0.25717,0.21411,0.18531,0.16672,0.15271,0.13753,0.12026,0.10875,0.09770,0.08605,0.07923,0.06919,0.06037,0.05439,0.04747,0.04035,0.03405,0.02572,0.01094]
    
    largest_k_stat_diff =  np.array(largest_k_stat_diff)

    highlight_fprs = [0, 0.05, 0.1, 0.2]
    overlap_results = {}
    j = 0
    for fpr_target in highlight_fprs:
        tp_indices_abs, fp_indices_abs, tp_images_abs, fp_images_abs = get_tp_fp(y_true, abs_median_diff, x_axis_medt[j], image_names)
        tp_indices_kstat, fp_indices_kstat, tp_images_kstat, fp_images_kstat = get_tp_fp(y_true, largest_k_stat_diff, x_axis_kst[j], image_names)

        overlap_tp_indices = list(set(tp_indices_abs).intersection(tp_indices_kstat))
        overlap_tp_images = [image_names[i] for i in overlap_tp_indices]
        
        unique_tp_abs = list(set(tp_indices_abs) - set(tp_indices_kstat))
        unique_tp_abs_images = [image_names[i] for i in unique_tp_abs]
        
        unique_tp_kstat = list(set(tp_indices_kstat) - set(tp_indices_abs))
        unique_tp_kstat_images = [image_names[i] for i in unique_tp_kstat]

        images_unique_detections = list(set(tp_images_abs + tp_images_kstat))

        overlap_results[f"{int(fpr_target * 100)}% FPR"] = {
            "Correctly detected by K-S": len(tp_indices_kstat),
            "Correctly detected by Abs Median": len(tp_indices_abs),
            "Total unique detections": {
                "Count": len(images_unique_detections),
                "Images": images_unique_detections
            },
            "Overlap": {
                "Count": len(overlap_tp_indices),
                "Images": overlap_tp_images,
            },
            "Unique to K-S": {
                "Count": len(unique_tp_kstat),
                "Images": unique_tp_kstat_images,
            },
            "Unique to Abs Median": {
                "Count": len(unique_tp_abs),
                "Images": unique_tp_abs_images,
            }
        }

        j += 1


    #save the overlap results to a JSON file
    with open(result_path + 'overlap_results.json', 'w') as file:
        json.dump(overlap_results, file, indent=4)


    #----------------------------------------------------------------------------------
    #plot the ROC curves
    #calculate positives for each threshold & normalize
    positives_above_threshold_median = [(np.sum(abs_median_diff > x)/ 201) for x in x_axis_medt ]
    positives_above_threshold_ks_stat = [(np.sum(largest_k_stat_diff > x)/ 201) for x in x_axis_kst ]


    #highlight TPR for specific values
    threshold_values = [0, 1, 2, 4] #indexing for FPR values of interest
    FPR_values = [0.00,0.05,0.1,0.2]
    tpr_abs_highlight = [positives_above_threshold_median[tval] for tval in threshold_values]
    tpr_kstat_highlight = [positives_above_threshold_ks_stat[tval] for tval in threshold_values]

    excpected_AUC_abs = estimate_auc(np.linspace(0.00, 1.00, 21),positives_above_threshold_median)
    excpected_AUC_kstat = estimate_auc(np.linspace(0.00, 1.00, 21),positives_above_threshold_ks_stat)


    plt.figure(figsize=(10, 8))
    plt.plot(np.linspace(0.00, 1.00, 21), positives_above_threshold_median, color='blue', lw=2, label=f'Absolute Median Difference (Expected AUC = {excpected_AUC_abs:.2f})')
    plt.plot(np.linspace(0.00, 1.00, 21), positives_above_threshold_ks_stat, color='magenta', lw=2, label=f'K-S Stat Difference (Expected AUC = {excpected_AUC_kstat:.2f})')
    plt.plot([1, 0], [1, 0], color='gray', lw=2, linestyle='--')

    color_markers = ['green', 'gold', 'darkorange', "red"]

    #highlight TPR for specific FPR values
    for i, fpr in enumerate(FPR_values):
        plt.plot(fpr, tpr_abs_highlight[i], 'o', color=color_markers[i],label=f'Median difference ({fpr:.2f}, {tpr_abs_highlight[i]:.2f})')
        plt.plot(fpr, tpr_kstat_highlight[i], 'd', color=color_markers[i], label=f'K-S statistic ({fpr:.2f}, {tpr_kstat_highlight[i]:.2f})')



    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Threshold FPR IMD2020')
    plt.ylabel('True Positives %')
    plt.title( 'In the Wild: ' + algo_name+', Expected TPR and AUC when relaxing Threshold')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(result_path + "ROC_curve_" + algo + "_" + dataset + ".png")



#calculate overlap/distinctons for MM-fusion-------------------------------------------------------------------

#start wild
mmwild = "results_MM-fusion_in_the_wild.json"
print(mmwild)
parts = mmwild.split("_")
last = len(parts)-1
dataset = parts[last][:-5]
algo = parts[1]
result_path = "results/" + algo + "/" + dataset

algo_name = "MM-Fusion"
    

#add slash so filename can be easily added
result_path = result_path + "/"

#load the JSON data from a file
with open(mmwild, 'r') as file:
    raw_data = file.read()

#fix the formatting by wrapping the objects in an array
raw_data = '[\n' + raw_data.replace('}\n{', '},\n{') + ']'

#parse the fixed JSON data
data = json.loads(raw_data)

#extract the necessary values
y_true = [1 if d["type"] == "manipulated" else 0 for d in data]
abs_median_diff = [d["absolute median difference"] for d in data]
largest_k_stat_diff = [d["k stat diff"] for d in data]
detection_scores_AI = [d["detection score AI-model"] for d in data]
image_names = [d["image_name"] for d in data]

#probability thresholds from IMD2020, from 0% FPR to 100FPR in steps of 0.05
x_axis_medt = [255.0,66.3,23.6,11.99,8.04,4.77,3.43,1.97,1.6,1.23,0.96,0.87,0.77,0.67,0.58,0.48,0.39,0.29,0.19,0.1,0.00]
x_axis_kst = [0.84,0.53,0.44,0.37,0.34,0.30,0.28,0.26,0.23,0.20,0.18,0.16,0.15,0.13,0.11,0.09,0.07,0.05,0.03,0.02,0.00]
x_axis_ait = [7.03,1.57,0.93,0.42,0.22,0.05,-0.03,-0.1,-0.14,-0.2,-0.29,-0.34,-0.41,-0.48,-0.52,-0.57,-0.66,-0.75,-0.83,-0.98,-2.12]


highlight_fprs = [0, 0.05, 0.1, 0.2]
overlap_results = {}
j = 0
for fpr_target in highlight_fprs:
    tp_indices_abs, fp_indices_abs, tp_images_abs, fp_images_abs = get_tp_fp(y_true, abs_median_diff, x_axis_medt[j], image_names)
    tp_indices_kstat, fp_indices_kstat, tp_images_kstat, fp_images_kstat = get_tp_fp(y_true, largest_k_stat_diff, x_axis_kst[j], image_names)
    tp_indices_aidet, fp_indices_aidet, tp_images_aidet, fp_images_aidet = get_tp_fp(y_true, detection_scores_AI, x_axis_ait[j], image_names)

    overlap_tp_indices_ai_ksmed = list(set(tp_indices_aidet).intersection(set(tp_indices_kstat + tp_indices_abs)))
    overlap_tp_images = [image_names[i] for i in overlap_tp_indices_ai_ksmed]

    unique_tp_abs = list(set(tp_indices_abs) - set(tp_indices_kstat) - set(tp_indices_aidet))
    unique_tp_abs_images = [image_names[i] for i in unique_tp_abs]

    unique_tp_kstat = list(set(tp_indices_kstat) - set(tp_indices_abs) - set(tp_indices_aidet))
    unique_tp_kstat_images = [image_names[i] for i in unique_tp_kstat]

    unique_tp_aidet = list(set(tp_indices_aidet) - set(tp_indices_abs) - set(tp_indices_kstat))
    unique_tp_aidet_images = [image_names[i] for i in unique_tp_aidet]

    images_unique_detections = list(set(tp_images_abs + tp_images_kstat + tp_images_aidet))

    overlap_results[f"{int(fpr_target * 100)}% FPR"] = {
        "Correctly detected by K-S": len(tp_indices_kstat),
        "Correctly detected by Abs Median": len(tp_indices_abs),
        "Correctly detected by AI Detection": len(tp_indices_aidet),
        "Total unique detections": {
            "Count": len(images_unique_detections),
            "Images": images_unique_detections
        },
        "Overlap AI and (k-s + med)": {
            "Count": len(overlap_tp_indices_ai_ksmed),
            "Images": overlap_tp_images,
        },
        "Unique to K-S": {
            "Count": len(unique_tp_kstat),
            "Images": unique_tp_kstat_images,
        },
        "Unique to Abs Median": {
            "Count": len(unique_tp_abs),
            "Images": unique_tp_abs_images,
        },
        "Unique to AI Detection": {
            "Count": len(unique_tp_aidet),
            "Images": unique_tp_aidet_images,
        }
    }

    j += 1

#save the overlap results to a JSON file
with open(result_path + 'overlap_results.json', 'w') as file:
    json.dump(overlap_results, file, indent=4)

#----------------------------------------------------------------------------------------------------------------

files_mmfusion = ["results_MM-fusion_IMD2020.json","results_MM-fusion_IFS.json","results_MM-fusion_Coverage2.json",
                   "results_MM-fusion_Coverage.json","results_MM-fusion_Columbia.json","results_MM-fusion_CocoGlide.json"]

for filename in files_mmfusion:
    print(filename)
    parts = filename.split("_")
    last = len(parts)-1
    dataset = parts[last][:-5]
    algo = parts[1]
    result_path = "results/" + algo + "/" + dataset

    algo_name = "MM-Fusion"
        

    #add slash so filename can be easily added
    result_path = result_path + "/"

    #load the JSON data from a file
    with open(filename, 'r') as file:
        raw_data = file.read()

    #fix the formatting by wrapping the objects in an array
    raw_data = '[\n' + raw_data.replace('}\n{', '},\n{') + ']'

    #parse the fixed JSON data
    data = json.loads(raw_data)

    #extract the necessary values
    y_true = [1 if d["type"] == "manipulated" else 0 for d in data]
    abs_median_diff = [d["absolute median difference"] for d in data]
    largest_k_stat_diff = [d["k stat diff"] for d in data]
    detection_scores_AI = [d["detection score AI-model"] for d in data]
    image_names = [d["image_name"] for d in data]

    #calculate ROC for absolute median difference
    fpr_abs, tpr_abs, thresholds_abs = roc_curve(y_true, abs_median_diff)
    roc_auc_abs = auc(fpr_abs, tpr_abs)

    #calculate ROC for largest k stat diff
    fpr_kstat, tpr_kstat, thresholds_kstat = roc_curve(y_true, largest_k_stat_diff)
    roc_auc_kstat = auc(fpr_kstat, tpr_kstat)

    #calculate ROC for AI detection score
    fpr_aidet, tpr_aidet, thresholds_aidet = roc_curve(y_true, detection_scores_AI)
    roc_auc_aidet = auc(fpr_aidet, tpr_aidet)

    highlight_fprs = [0, 0.05, 0.1, 0.2]
    overlap_results = {}

    for fpr_target in highlight_fprs:
        threshold_abs = find_threshold_at_fpr(fpr_abs, thresholds_abs, fpr_target)
        threshold_kstat = find_threshold_at_fpr(fpr_kstat, thresholds_kstat, fpr_target)
        threshold_aidet = find_threshold_at_fpr(fpr_aidet, thresholds_aidet, fpr_target)

        tp_indices_abs, fp_indices_abs, tp_images_abs, fp_images_abs = get_tp_fp(y_true, abs_median_diff, threshold_abs, image_names)
        tp_indices_kstat, fp_indices_kstat, tp_images_kstat, fp_images_kstat = get_tp_fp(y_true, largest_k_stat_diff, threshold_kstat, image_names)
        tp_indices_aidet, fp_indices_aidet, tp_images_aidet, fp_images_aidet = get_tp_fp(y_true, detection_scores_AI, threshold_aidet, image_names)

        overlap_tp_indices_ai_ksmed = list(set(tp_indices_aidet).intersection(set(tp_indices_kstat + tp_indices_abs)))
        overlap_tp_images = [image_names[i] for i in overlap_tp_indices_ai_ksmed]

        unique_tp_abs = list(set(tp_indices_abs) - set(tp_indices_kstat) - set(tp_indices_aidet))
        unique_tp_abs_images = [image_names[i] for i in unique_tp_abs]

        unique_tp_kstat = list(set(tp_indices_kstat) - set(tp_indices_abs) - set(tp_indices_aidet))
        unique_tp_kstat_images = [image_names[i] for i in unique_tp_kstat]

        unique_tp_aidet = list(set(tp_indices_aidet) - set(tp_indices_abs) - set(tp_indices_kstat))
        unique_tp_aidet_images = [image_names[i] for i in unique_tp_aidet]

        overlap_fp_indices_ai_ksmed = list(set(fp_indices_aidet).intersection(set(fp_indices_kstat+fp_indices_abs)))
        overlap_fp_images = [image_names[i] for i in overlap_fp_indices_ai_ksmed]

        unique_fp_abs = list(set(fp_indices_abs) - set(fp_indices_kstat) - set(fp_indices_aidet))
        unique_fp_abs_images = [image_names[i] for i in unique_fp_abs]

        unique_fp_kstat = list(set(fp_indices_kstat) - set(fp_indices_abs) - set(fp_indices_aidet))
        unique_fp_kstat_images = [image_names[i] for i in unique_fp_kstat]

        unique_fp_aidet = list(set(fp_indices_aidet) - set(fp_indices_abs) - set(fp_indices_kstat))
        unique_fp_aidet_images = [image_names[i] for i in unique_fp_aidet]

        
        images_unique_detections = list(set(tp_images_abs + tp_images_kstat + tp_images_aidet))

        total_false_positives_images = list(set(fp_images_abs + fp_images_kstat + fp_images_aidet))

        overlap_results[f"{int(fpr_target * 100)}% FPR"] = {
            "Correctly detected by K-S": len(tp_indices_kstat),
            "Correctly detected by Abs Median": len(tp_indices_abs),
            "Correctly detected by AI Detection": len(tp_indices_aidet),
            "Total unique detections": {
                "Count": len(images_unique_detections),
                "Images": images_unique_detections
            },
            "Overlap AI and (k-s + med)": {
                "Count": len(overlap_tp_indices_ai_ksmed),
                "Images": overlap_tp_images,
            },
            "Unique to K-S": {
                "Count": len(unique_tp_kstat),
                "Images": unique_tp_kstat_images,
            },
            "Unique to Abs Median": {
                "Count": len(unique_tp_abs),
                "Images": unique_tp_abs_images,
            },
            "Unique to AI Detection": {
                "Count": len(unique_tp_aidet),
                "Images": unique_tp_aidet_images,
            },
            "False positives by K-S": len(fp_indices_kstat),
            "False positives by Abs Median": len(fp_indices_abs),
            "False positives by AI Detection": len(fp_indices_aidet),
            "Total False positives": {
                "Count": len(total_false_positives_images),
                "Images": total_false_positives_images,
            },
            "False Positives": {
                "Overlap AI and (k-s + med)": {
                    "Count": len(overlap_fp_indices_ai_ksmed),
                    "Images": overlap_fp_images,
                },
                "Unique to K-S": {
                    "Count": len(unique_fp_kstat),
                    "Images": unique_fp_kstat_images,
                },
                "Unique to Abs Median": {
                    "Count": len(unique_fp_abs),
                    "Images": unique_fp_abs_images,
                },
                "Unique to AI Detection": {
                    "Count": len(unique_fp_aidet),
                    "Images": unique_fp_aidet_images,
                },
            }
        }

    #save the overlap results to a JSON file
    with open(result_path + 'overlap_results.json', 'w') as file:
        json.dump(overlap_results, file, indent=4)




#----------------------------------------------------------------------------------------------------------------------
#calculate overlap classical techniques and mm-fusion

def get_json_overlap(filename):
    parts = filename.split("_")
    last = len(parts)-1
    dataset = parts[last][:-5]
    algo = parts[1]
    overlap_path = "results/" + algo + "/" + dataset + "/overlap_results.json"

    with open(overlap_path, 'r') as file:
        data = json.load(file)

    return data


highlight_fprs = [0, 0.05, 0.1, 0.2]
all_datsets = datasets_orig_manip + [wild]
all_mmfusion_files = files_mmfusion + [mmwild]
for dataset in all_datsets: 
    
    filename_dataset = dataset[0]
    parts = filename_dataset.split("_")
    last = len(parts)-1
    dataset_name = parts[last][:-5]
    print(dataset_name)
    for fusionfile in all_mmfusion_files:
        if dataset_name in fusionfile:
            mmfusion_file = fusionfile
    
    ai_overlap_data = get_json_overlap(mmfusion_file)


    overlap_results = {}
    for fpr_target in highlight_fprs:
        FPR_key = f"{int(fpr_target * 100)}% FPR"
        total_unique_detected_classical_images = set() #only K-S statistic, including the med diff would be unfair because the false positives of both techniques do not overlap
        for file in dataset:
            overlap_data = get_json_overlap(file)
            total_unique_detected_classical_images.update(overlap_data[FPR_key]["Overlap"]['Images'])
            total_unique_detected_classical_images.update(overlap_data[FPR_key]["Unique to K-S"]['Images'])
        
        total_unique_detected_classical = len(total_unique_detected_classical_images)

        total_unique_detected_ai_images = set(ai_overlap_data[FPR_key]["Overlap AI and (k-s + med)"]['Images'])
        total_unique_detected_ai_images.update(ai_overlap_data[FPR_key]["Unique to AI Detection"]['Images'])
        total_unique_detected_ai = len(total_unique_detected_ai_images)
        
        images_unique_detections = total_unique_detected_classical_images.union(total_unique_detected_ai_images)
        overlap_images = total_unique_detected_classical_images.intersection(total_unique_detected_ai_images)

        unique_classical_images = list(set(total_unique_detected_classical_images) - set(total_unique_detected_ai_images))
        unique_ai_images = list(set(total_unique_detected_ai_images) - set(total_unique_detected_classical_images))
        
        if dataset_name == "wild":
            #no false positive in wild dataset
            fp_total_unique_detected_classical = 0
            fp_total_unique_detected_ai= 0
            fp_images_unique_detections = []
            fp_overlap_images = []
            fp_unique_classical_images = []
            fp_unique_ai_images = []
        else:
            #false positives overlap
            fp_total_unique_detected_classical_images = set()
            for file in dataset:
                overlap_data = get_json_overlap(file)
                #only K-S statistic
                fp_total_unique_detected_classical_images.update(overlap_data[FPR_key]["False Positives"]["Overlap"]['Images'])
                fp_total_unique_detected_classical_images.update(overlap_data[FPR_key]["False Positives"]["Unique to K-S"]['Images'])
            
            fp_total_unique_detected_classical = len(fp_total_unique_detected_classical_images)
            
            fp_total_unique_detected_ai_images = set(ai_overlap_data[FPR_key]["False Positives"]["Overlap AI and (k-s + med)"]['Images'])
            fp_total_unique_detected_ai_images.update(ai_overlap_data[FPR_key]["False Positives"]["Unique to AI Detection"]['Images'])
            fp_total_unique_detected_ai = len(fp_total_unique_detected_ai_images)
            
            fp_images_unique_detections = fp_total_unique_detected_classical_images.union(fp_total_unique_detected_ai_images)
            fp_overlap_images = fp_total_unique_detected_classical_images.intersection(fp_total_unique_detected_ai_images)
            
            fp_unique_classical_images = list(fp_total_unique_detected_classical_images - fp_total_unique_detected_ai_images)
            fp_unique_ai_images = list(fp_total_unique_detected_ai_images - fp_total_unique_detected_classical_images)

        overlap_results[f"{int(fpr_target * 100)}% FPR"] = {
            "Correctly detected by classical techniques (K-S)": total_unique_detected_classical,
            "Correctly detected by MM-fusion (AI)": total_unique_detected_ai,
            "Total unique detections": {
                "Count": len(images_unique_detections),
                "Images": list(images_unique_detections)
            },
            "Overlap AI and classical": {
                "Count": len(overlap_images),
                "Images": list(overlap_images),
            },
            "Unique to classical": {
                "Count": len(unique_classical_images),
                "Images": list(unique_classical_images),
            },
            "Unique to AI Detection": {
                "Count": len(unique_ai_images),
                "Images": list(unique_ai_images),
            },
            "False positives by classical techniques (K-S)": fp_total_unique_detected_classical,
            "False positives detected by MM-fusion (AI)": fp_total_unique_detected_ai,
            "False positives: Total unique detections": {
                "Count": len(fp_images_unique_detections),
                "Images": list(fp_images_unique_detections)
            },
            "False positives: Overlap AI and classical": {
                "Count": len(fp_overlap_images),
                "Images": list(fp_overlap_images),
            },
            "False positives: Unique to classical": {
                "Count": len(fp_unique_classical_images),
                "Images": fp_unique_classical_images,
            },
            "False positives: Unique to AI Detection": {
                "Count": len(fp_unique_ai_images),
                "Images": fp_unique_ai_images,
            }
        }

    #save
    with open(dataset_name + '_overlap_results_AI_classical.json', 'w') as file:
        json.dump(overlap_results, file, indent=4)

