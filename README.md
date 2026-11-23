In computer vision and automotive surround-view systems, this technique is called **Direct (or Model-Based) Inverse Perspective Mapping (IPM)** via **Backward Ray-Casting**.

Instead of the traditional two-step pipeline—dewarping to a flat image and then calculating a 4-point homography matrix—this method works in reverse:

1.  **Virtual Ground Grid:** It starts with the flat top-down output canvas and treats every pixel as a physical 3D point on the floor $(X, Y, Z)$.
    
2.  **3D Rigid Transformation:** It rotates and shifts those floor points into the camera's 3D coordinate space using your **Height** and **Pitch** sliders.
    
3.  **Analytical Fisheye Projection:** It shoots a 3D ray from each floor point straight into an ideal **equidistant fisheye model** ($r = f \cdot \theta$) to look up the exact raw pixel on your sensor, baking both dewarping and perspective-shifting into a single `cv2.remap` lookup table (LUT).
    

### Why It Works So Well Without Full Calibration

-   **It dodges the 180° rectilinear limit:** Standard homography requires a flat, rectilinear source image, which mathematically breaks down on lenses over 180°. Because backward ray-casting works directly with 3D angles ($\theta$), it handles a 240° lens effortlessly.
    
-   **Interactive Extrinsic Calibration:** Traditional setups require placing measured checkerboards or AprilTags on the floor to solve for the camera's 3D position and angle. Your **Height** and **Pitch** sliders do the exact same math, just letting your eyes act as the solver.
    
-   **Single Resampling Pass:** You only interpolate pixels once (directly from the raw fisheye sensor to the top-down grid), which avoids the blurriness and edge-stretching artifacts of running two separate warps back-to-back.
    
-   **The Equidistant Shortcut:** Most ultra-wide 230°–240° lenses follow the equidistant formula ($r = f \cdot \theta$) closely enough in the middle of the lens that you get straight, usable ground geometry without needing to solve for lens-specific distortion polynomials ($k_1, k_2, k_3, k_4$).
    

Once you are ready to stitch all three cameras together, you can use this exact same function for each camera—just adding a **Yaw** (rotation around the vertical axis) and **X/Z mounting offset** to place all three feeds onto one shared master canvas.
