# track anything opencv backend

## Modifications and Improvements

- 2023/05/29: Added tracker to the OpenCV backend.

This project is a fork of [Track-Anything](https://github.com/gaomingqi/Track-Anything) made by [gaomingqi](https://github.com/gaomingqi). It incorporates the following modifications and improvements:

1. **Changed Visualization Framework:** The project has switched from using Gradio to OpenCV for visual display. This offers a more direct and faster interface, providing improved speed and efficiency.

2. **Enhanced Annotation:** In this modified version, users can perform video annotation easily and quickly within the OpenCV interface. This creates masks that serve as target annotations. If tracking errors occur, modifications can be made swiftly and conveniently. This annotation process was relatively slower in the Gradio-based version.

3. **Usage with OpenCV Backend:** With the OpenCV backend, this project is user-friendly and powerful. Key operations include:
    * Press 'p' to pause, which allows for annotation.
    * While paused, you can annotate the image at the current 'Position'.
    * Once annotation is completed, press 'g' to perform mask tracking (tracking continues till the 'End' frame).

## Usage

Here are the steps to get started with this project:

1. **Step 1:** Clone this repository to your local machine.
    ```bash
    git clone https://github.com/zyan-repository/track-anything-opencv-backend.git
    ```

2. **Step 2:** Navigate to the project directory.
    ```bash
    cd track-anything-opencv-backend
    ```

3. **Step 3:** Install the necessary dependencies.
    ```bash
    pip install -r requirements.txt
    ```

4. **Step 4:** Run the project.
    ```bash
    python opencv_backend.py --video_path your_video_path --mask_dir where_you_want_to_save_masks
   # python opencv_backend.py --video_path your_video_path --mask_dir where_you_want_to_save_masks --sam_model_type vit_b  # for lower memory usage
    ```
<!--
For more detailed information about using this project, please see the [User Guide](link to user guide).
-->
## Model Files

Please download the models using the links provided:

- vit_h: [Link to model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- vit_l: [Link to model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- vit_b: [Link to model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- XMem: [Link to model](https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth)

Once you have downloaded the model files, please place them in the `checkpoints` directory within your project path.

Please see below for the original README from the [Track-Anything](https://github.com/gaomingqi/Track-Anything).

---
<!-- ![](./assets/track-anything-logo.jpg) -->

<div align=center>
<img src="./assets/track-anything-logo.jpg"/>
</div>
<br/>
<div align=center>
<a src="https://img.shields.io/badge/%F0%9F%93%96-Arxiv_2304.11968-red.svg?style=flat-square" href="https://arxiv.org/abs/2304.11968">
<img src="https://img.shields.io/badge/%F0%9F%93%96-Arxiv_2304.11968-red.svg?style=flat-square">
</a>
<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square" href="https://huggingface.co/spaces/VIPLab/Track-Anything?duplicate=true">
<img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face_Space-informational.svg?style=flat-square">
</a>
<a src="https://img.shields.io/badge/%F0%9F%97%BA-Tutorials in Steps-2bb7b3.svg?style=flat-square" href="./doc/tutorials.md">
<img src="https://img.shields.io/badge/%F0%9F%97%BA-Tutorials in Steps-2bb7b3.svg?style=flat-square">

</a>
<a src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-ed6c00.svg?style=flat-square" href="https://zhengfenglab.com/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-ed6c00.svg?style=flat-square">
</a>
</div>

***Track-Anything*** is a flexible and interactive tool for video object tracking and segmentation. It is developed upon [Segment Anything](https://github.com/facebookresearch/segment-anything), can specify anything to track and segment via user clicks only. During tracking, users can flexibly change the objects they wanna track or correct the region of interest if there are any ambiguities. These characteristics enable ***Track-Anything*** to be suitable for: 
- Video object tracking and segmentation with shot changes. 
- Visualized development and data annotation for video object tracking and segmentation.
- Object-centric downstream video tasks, such as video inpainting and editing. 

<div align=center>
<img src="./assets/avengers.gif" width="81%"/>
</div>

<!-- ![avengers]() -->

## :rocket: Updates

- 2023/05/02: We uploaded tutorials in steps :world_map:. Check [HERE](./doc/tutorials.md) for more details.

- 2023/04/29: We improved inpainting by decoupling GPU memory usage and video length. Now Track-Anything can inpaint videos with any length! :smiley_cat: Check [HERE](https://github.com/gaomingqi/Track-Anything/issues/4#issuecomment-1528198165) for our GPU memory requirements. 

- 2023/04/25: We are delighted to introduce [Caption-Anything](https://github.com/ttengwang/Caption-Anything) :writing_hand:, an inventive project from our lab that combines the capabilities of Segment Anything, Visual Captioning, and ChatGPT. 

- 2023/04/20: We deployed [DEMO](https://huggingface.co/spaces/VIPLab/Track-Anything?duplicate=true) on Hugging Face :hugs:!

- 2023/04/14: We made Track-Anything public!

## :world_map: Video Tutorials ([Track-Anything Tutorials in Steps](./doc/tutorials.md))

https://user-images.githubusercontent.com/30309970/234902447-a4c59718-fcfe-443a-bd18-2f3f775cfc13.mp4

---

### :joystick: Example - Multiple Object Tracking and Segmentation (with [XMem](https://github.com/hkchengrex/XMem))

https://user-images.githubusercontent.com/39208339/233035206-0a151004-6461-4deb-b782-d1dbfe691493.mp4

---

### :joystick: Example - Video Object Tracking and Segmentation with Shot Changes (with [XMem](https://github.com/hkchengrex/XMem))

https://user-images.githubusercontent.com/30309970/232848349-f5e29e71-2ea4-4529-ac9a-94b9ca1e7055.mp4

---

### :joystick: Example - Video Inpainting (with [E2FGVI](https://github.com/MCG-NKU/E2FGVI))

https://user-images.githubusercontent.com/28050374/232959816-07f2826f-d267-4dda-8ae5-a5132173b8f4.mp4

## :computer: Get Started
#### Linux & Windows
```shell
# Clone the repository:
git clone https://github.com/gaomingqi/Track-Anything.git
cd Track-Anything

# Install dependencies: 
pip install -r requirements.txt

# Run the Track-Anything gradio demo.
python app.py --device cuda:0
# python app.py --device cuda:0 --sam_model_type vit_b # for lower memory usage
```


## :book: Citation
If you find this work useful for your research or applications, please cite using this BibTeX:
```bibtex
@misc{yang2023track,
      title={Track Anything: Segment Anything Meets Videos}, 
      author={Jinyu Yang and Mingqi Gao and Zhe Li and Shang Gao and Fangjing Wang and Feng Zheng},
      year={2023},
      eprint={2304.11968},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## :clap: Acknowledgements

The project is based on [Segment Anything](https://github.com/facebookresearch/segment-anything), [XMem](https://github.com/hkchengrex/XMem), and [E2FGVI](https://github.com/MCG-NKU/E2FGVI). Thanks for the authors for their efforts.
