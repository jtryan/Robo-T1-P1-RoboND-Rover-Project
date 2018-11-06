## Project: Search and Sample Return

---

**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your Rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[settings]: ./calibration_images/simulator-settings.png
[warped]: ./calibration_images/warped.png
[thresh]: ./calibration_images/thresh.png
[rock]: ./calibration_images/rock.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
 
---
### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

I began this project by cloning it from the github repository. After activating the RoboND python environment I ran `jupyter notebook`, then opened the `Rover_Project_Test_Notebook`. I ran the simulator in OS X using the following settings.

![alt text][settings]

I further ran all of my autonomous testing using the same settings.

### Test Notebook

Using the simulator I collected some test data that could be used for building a pipeline that would process the images and allow the Rover to act autonomously.

One of the first thins that mut be done is to calibrate the images from the camera. This process takes an image from the camera and transforms it so that is is displayed from a top down perspective. The python library `numpy` has built in methods that allows one to do this very simply. The follwing code snippet shows the code that takes an image and performs the transformation function.

```python
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0])) # keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask
```

After the function is defined source image and destination image sizes are defined and entered into the function for a result. The before image has a grid overlayed which made it easier to select 4 points on the image. 

Before Image

![alt text][image2]

The image after it has been transformed. Notice the lighter areas are where the Rover can navigate.

![alt text][warped]

After camera calibration I was ready to identify rocka and obstacles. The Rover uses a camera to navigate. To help the Rover identify obstacles and rocks I used color thresholds on the images which cleary delineated where navigateable terrain and rocks occur.  This code snippet show the process for seperating obstacles and terrain. I applied a color threshold in a range of 0-255 across all three color channels (Red, Green, Blue). The resulting image is then rendered in black and white, with Black being an osbstacle.

```python
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
```

Here is the image after  the code was applied.

![alt text][thresh]

To find rocks in the image a similar function was used with different numbers. Since the rock samples are mostly yellow focus was placed on filtering out green and blue.

```python
def find_rocks(img, rgb_thresh=(110, 110, 50)):
    color_select = np.zeros_like(img[:,:,0])
    rock_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])

    color_select[rock_thresh] = 1

    return color_select
```

### process_image()

I built the process_image() function following the information under the TODO section of the code. I first populated some variables pulling data from the image.

```python
xpos = data.xpos[data.count]
ypos = data.ypos[data.count]
yaw = data.yaw[data.count]
scale = 2 * dst_size
world_size = data.worldmap.shape[0]
```

The source and destination variables were available from a previuos function so they were used for the next step in the function, transforming the image and applying the color thresholds.

```python
dst_size = 5 
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],

warped, mask = perspect_transform(img, source, destination)
threshed = color_thresh(warped)
```

An obstacle map was then created from the threshed image. After converting the image coordinates to Rover coordinates, world coordinates were created with the pix_to_world function. The same was done with the obstacle coordinates.

```python {.line-numbers}
obstacle_map = np.absolute(np.float32(threshed) -1) * mask
xpix, ypix = Rover_coords(threshed)
x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
obs_xpix, obs_ypix = Rover_coords(obstacle_map)
obstacle_x_world, obstacle_y_world = pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)
```

Rock samples were found in a similar fashion using this bit of code.

```python
rock_map = find_rocks(warped, rgb_thresh=(110, 110, 50))
if rock_map.any():
    rock_x, rock_y = Rover_coords(rock_map)
    rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw, world_size, scale)
```

Finally the world map was updated using the world coordinates for the Rover, obstacles and samples. They were each given a channel on the world map. Red for the obstacles, blue for the Rover and all channels for the samples. The worldmap was updated using the following code.

```python
data.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255 # red channel [+=1]
data.worldmap[y_world,  x_world, 2] = 255 # blue channel
data.worldmap[rock_y_world, rock_x_world, :] = 255
```

The code to make the output image was in place and not changed. The output image was returned by the function.

To make the movie, existing code was used which ran the in the function and output it to a video file. That file is viewable in the `output` directory.

```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip


# Define pathname to save the output video
output = '../output/test_mapping.mp4'
data = Databucket() # Re-initialize data in case you're running this cell multiple times
clip = ImageSequenceClip(data.images, fps=60) # Note: output video will be sped up because 
                                          # recording rate in simulator is fps=25
new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
%time new_clip.write_videofile(output, audio=False)
```

### Autonomous Navigation and Mapping

#### perception_step()

In the `perception.py` module I had to build the `perception_step()` function. The work from the `Rover_Project_Test_Notebook` was available to place into the file with some changes. Instead of reading from a `data` file I was now reding working with a Rover object which contained an image in it state. I added an image variable into the function `image = Rover.img`.

Driving the vehicle autonomously requires steering angles to be sent to the Rover. To obtain the angle for the general direction of the Rover the x,y coordinates need to be converted to ppolar coordinates to obtain a vecotr with a distance and angle. The angle is then added to the Rover state. The follwing snippets shows this.

```python
dist, angles = to_polar_coords(xpix, ypix)
Rover.nav_angles = angles
```

The code for finding rocks was sufficient for identifying them and placing them on the map, but I watned to try and collect them. So I added a state called 'sample_found' which I would set if a sample was located. A vector to the sample was calculated which was set on the Rover. The pertinent code follows.

```python
rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
Rover.mode = 'sample_found'
Rover.nav_dists = rock_dist
Rover.nav_angles = rock_ang
```

#### decision_step()

The `decision.py` module has the `decision_step()` function. Here I added commands for the Rover while the rock was found yet not retrieved. 

```python
elif Rover.mode == 'sample_found':
    if Rover.vel < Rover.max_vel:
        Rover.throttle = Rover.throttle_set
    else:
        Rover.throttle = 0
    Rover.brake = 0
    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
    if Rover.near_sample:
        Rover.brake = Rover.brake_set
        Rover.mode = 'stop'
```

While the Rover was in the 'sample_found' mode it would accelrate and steer toward the sample. Once it was close enough to the sample to be picked up it received a state chage form another function that set `Rover.near_sample` to 1. Then the Rover would brake and pick up the sample. Its mode would be changed to 'stop' and it would continue. 

I noticed that retreiving sample would usually place the Rover into a position where it would get stuck, thinking it could turn away from the wall but it couldn't. So I added a conditional to check if it wasn't changing postion while in 'forward' mode. If it was True then the Rover would stop and try to turn with all four wheels.

```python
# Am I stuck?
if Rover.prev_pos == Rover.pos:
    Rover.throttle = 0
    Rover.brake = 0
    Rover.steer = -15 if np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15) < 0 else 15
```

The final line of code added a left or right decision to steering. I replaced the steering code in this secton also.

```python
if len(Rover.nav_angles) < Rover.go_forward:
    Rover.throttle = 0
    # Release the brake to allow turning
    Rover.brake = 0
    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
    # Rover.steer = -15 # Could be more clever here about which way to turn
    if Rover.steer == 0:
        Rover.steer = -15 if np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15) < 0 else 15
    # If we're stopped but see sufficient navigable terrain in front then go!
```

## Final Thoughts

The project shows what you can do using just computer vision to control a robot. I had a passing project very quickly, but I was really interested in trying to pick up samples. So I spent a large amount of time trying different algorithms to steer to the sample and allowing the Rover to recover from being stuck. I also had trouble picking the right location in the code to add the fix. 

I used Intellij Pycharm to perform debugging while running in autonomous mode. It was helpful, but not being able to pause the simulator on hitting a breakpoint was frustrating. 

I was able to find and pick up all rocks, but did not implement the code to go back to the starting point. I think I would add a 'go-home' mode to do that. 

If I spend more time on this, I would work more on improving the decision tree. I would also work on how to improve mapping, amd ensure I was able to explore all terrain. There is a section at the top right corner of the map, which ahs a lot of empty space and a spur to the right. The Rover would not always epxlore it. ALso the large amount of open terrain confused the Rover and sent it into a circle. The only way I could fix it was to slow down the max velocity of the Rover so it had more time to make a decision. I also could not work out how to improve fidelity nor understand how it related to mapping, the difference between good-pixels and bad-pixels. I think I could eventually figure it out improving my mapping. Improved mapping would lead to better exploration.

![alt text][image1]


