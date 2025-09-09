# Music-to-Album-Cover: Genre & Emotion Driven Diffusion

This project classifies music by **genre** and **emotion**, then transforms the results into a prompt for a diffusion model to generate an album cover that visually matches the music’s style and mood.  

---

## 1. Features
- Classify music genre and emotion  
- Generate structured prompts  
- Create album covers via diffusion models  

---

## 2. Project Architecture

Input (music) → Classifier (genre & emotion) → Prompt → Diffusion Model → Album Cover

---

## 3. Demo

We experiment with different prompt templates, replacing `{a}` with **genre prediction** and `{b}` with **emotion prediction**:

### (1) A musician whose music style is {a} and {b}.
<img src="https://drive.google.com/uc?export=view&id=1d4E4_otNC7CNWVTJdfxTrlZOYOskKc9I" height="200"/>

- Assumption：Sentence is too long
  
### (2) A musician who is {a} and {b}.
<img src="https://drive.google.com/uc?export=view&id=1z9rLeG9BV4cBPzaFbSYLSQqYdr6aPUmC" height="200"/>

- Assumption：Can't understand 'musician'

### (3) A man who is {a} and {b}. → the best
<img src="https://drive.google.com/uc?export=view&id=10VIv1gwQ2jahgnPagXfAfQwGFJknKthg" height="200"/>

### (4) musician, {a}, {b}
<img src="https://drive.google.com/uc?export=view&id=1hVwli2V-piGOrnO6bv_nGP7RhJlPc3eG" height="200"/>

### (5) man, {a}, {b}
<img src="https://drive.google.com/uc?export=view&id=14hsLh2dsuEbJMen-AekSiu6HEU_yNonO" height="200"/>

### (6) man, music, {a}, {b}
<img src="https://drive.google.com/uc?export=view&id=1Mzy6efr1884DHvrztbWYDsDs8Q3-M9Le" height="200"/>

---

## 4. Diffusion Model Deployment
- Model: **small-stable-diffusion-v0**
<img src="https://drive.google.com/uc?export=view&id=1lYz-sNRUtliQT_Xz1bl9U1prE6MnUIJp" height="250"/>

- Deployed on: **NVIDIA Jetson Xavier**
