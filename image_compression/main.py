import cv2
import numpy

from utils          import *
from features       import *
from scoring        import *
from tiles_merging  import * 



def show_process(image, scores, rectangles, animate):
    width  = image.shape[0]
    height = image.shape[1]
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    scores = cv2.resize(scores, (width, height), interpolation=cv2.INTER_LINEAR)
    scores = numpy.expand_dims(scores, 2)

    blue_color = numpy.zeros((1, 1, 3))
    blue_color[0,0,0] = 1.0

    red_color  = numpy.zeros((1, 1, 3))
    red_color[0,0,2] = 1.0

    scores_colored = (1.0 - scores)*blue_color + scores*red_color

    

    image_tiles = numpy.array(image)
    image_tiles_plot = numpy.zeros_like(image)

    
    while True:
        image_tiles = numpy.array(image)
        image_tiles_plot = numpy.zeros_like(image)

        for x, y, w, h, score in rectangles:
            image_tiles = cv2.rectangle(image_tiles, (x, y), (x+w, y+h), (1, 0, 0), 3)

            image_tiles_plot[y:y+h, x:x+w, :] = image[y:y+h, x:x+w, :]
            
            top_row = np.hstack((image, scores_colored))
            bottom_row = np.hstack((image_tiles, image_tiles_plot)) 

            result = np.vstack((top_row, bottom_row))

            if animate:
                cv2.imshow("visualisation", result)
                key = cv2.waitKey(1)

                if key == 27:
                    return

        if animate != True:
            cv2.imshow("visualisation", result)
            cv2.imwrite("result.jpg", numpy.array(255*result, dtype=numpy.uint8))
            cv2.waitKey(0)
            return

        
    



if __name__ == "__main__":

    TILE_SIZE = 16

    #image = load_image("data/Comet_on_10_February_2016_NavCam.jpg")
    #image = load_image("data/Comet_on_15_April_2015_b_NavCam.jpg")
    #image = load_image("data/Comet_from_17.4_km_NavCam.jpg")
    #image = load_image("data/Comet_from_19.4_km_NavCam.jpg")
    image = load_image("data/Comet_from_20_km_NavCam.jpg")


    #image = load_image("data/Comet_on_15_April_2015_b_NavCam.jpg")
    #image = load_image("data/Comet_on_10_February_2016_NavCam.jpg")
    #image = load_image("data/Comet_from_19.4_km_NavCam.jpg")
 
    
   
    # Pipeline execution
    features = extract_tile_features_advanced(image, tile_size=TILE_SIZE)
    print("features = ", features.shape)


    #scores = score_tiles_iforest(features)
    scores = score_tiles(features)
    
    merged_rects = merge_tiles_q(scores, tile_size=TILE_SIZE, min_score_threshold=0.5, max_block_tiles=16)

    show_process(image, scores, merged_rects, True)
