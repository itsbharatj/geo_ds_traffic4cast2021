# Traffic4cast Dataset Structure and Format

## Dynamic Data (Traffic Data)

For each city, the dynamic data for a single day is represented as a tensor of shape `(288,495,436,8)` where:
* 288 represents 5-minute time bins (covering 24 hours)
* 495×436 is the spatial resolution (grid cells)
* 8 channels represent traffic volume and speed for four heading directions

Mathematically, we can represent this as:
$$X_{day} \in \mathbb{R}^{288 \times 495 \times 436 \times 8}$$

The 8 channels encode:
* Channels 0,1: Volume and speed for heading NE (0-90 degrees)
* Channels 2,3: Volume and speed for heading NW
* Channels 4,5: Volume and speed for heading SE
* Channels 6,7: Volume and speed for heading SW

## Static Data (Road Network)

Each city also has static data represented as:
1. `<CITY_NAME>_static.h5`: A tensor of shape `(9,495,436)` where:
   * Channel 0: Grayscale map of the city
   * Channels 1-8: Binary encoding of connections to neighboring cells (N, NE, E, SE, S, SW, W, NW)
2. `<CITY_NAME>_static_map_high_res.h5`: Higher resolution map (4950×4360)

Mathematically:
$$S \in \mathbb{R}^{9 \times 495 \times 436}$$

## Model Input-Output Structure

For the traffic forecasting task, the input to the model is a 12 × 495 × 436 × 8 tensor, where 495 × 436 is the spatial resolution of the city heatmap, and the 8 channels represent volume and speed for four headings (NE, SE, SW, and NW). The input consists of 12 consecutive heatmaps of 5-minute intervals, spanning a total of 1 hour.

For the test set, each file contains a tensor of size `(100,12,495,436,8)`:
* 100 represents the number of test instances
* 12 consecutive "images" of 5-minute intervals (1 hour total)
* 495×436 spatial resolution
* 8 channels (volume and speed for 4 headings)

The model needs to predict 6 future time steps: 5min, 10min, 15min, 30min, 45min, and 60min into the future. This corresponds to an output tensor of shape `(100,6,495,436,8)`.

Mathematically:
* Input: $$X_{in} \in \mathbb{R}^{100 \times 12 \times 495 \times 436 \times 8}$$
* Output: $$X_{out} \in \mathbb{R}^{100 \times 6 \times 495 \times 436 \times 8}$$

For a single sample:
* Input: $$x_{in} \in \mathbb{R}^{12 \times 495 \times 436 \times 8}$$
* Output: $$x_{out} \in \mathbb{R}^{6 \times 495 \times 436 \times 8}$$

In the paper's implementation, they concatenate the 12 heatmaps across the channel dimension, resulting in a tensor of shape 495 × 436 × 96. They additionally concatenate the static information of shape 495 × 436 × 9 to the input, giving a final input tensor of shape 495 × 436 × 105 to their U-Net model.

The U-Net then outputs a tensor of shape 495 × 436 × 48, which is reshaped into 6 × 495 × 436 × 8, corresponding to the 6 predicted time steps.