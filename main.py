import os
import cv2

# sample = cv2.imread("./SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")
sample = cv2.imread("./150__M_Right_index_finger_Obl_Custom.BMP")
sample = cv2.resize(sample, None, fx=2.5, fy=2.5)

# cv2.imshow('sample', sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

best_score = 0
file_name = None
image = None

kp1, kp2, mp = None, None, None

# os_list_dir = os.listdir("./SOCOFing/Real")
# print(os_list_dir)

counter = 0
for file in [file for file in os.listdir("./SOCOFing/Real")]:
    if counter % 100 == 0:
        print(counter)
    counter += 1
    finger_print_image = cv2.imread("./SOCOFing/Real/" + file)
    sift = cv2.SIFT_create()

    key_points_1, descriptions_1 = sift.detectAndCompute(sample, None)
    key_points_2, descriptions_2 = sift.detectAndCompute(finger_print_image, None)

    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(
        descriptions_1, descriptions_2, k=2
    )

    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    key_points = min(len(key_points_1), len(key_points_2))

    if len(match_points) / key_points * 100 > best_score:
        best_score = len(match_points) / key_points * 100
        image = finger_print_image
        kp1, kp2, mp = key_points_1, key_points_2, match_points
        file_name = file  # Gán giá trị cho file_name

if image is not None:
    print("best match: " + str(file_name))
    print("best score: " + str(best_score))

    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=2.5, fy=2.5)

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No match found.")
