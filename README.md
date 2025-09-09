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

### (3) A musician who is {a} and {b}.
<img src="https://drive.google.com/uc?export=view&id=10VIv1gwQ2jahgnPagXfAfQwGFJknKthg" height="200"/>

### (4) A man who is {a} and {b}.
<img src="https://drive.google.com/file/d/1hVwli2V-piGOrnO6bv_nGP7RhJlPc3eG/view?usp=share_link" height="200"/>

### (5) musician, {a}, {b}
<img src="https://drive.google.com/uc?export=view&id=1z9rLeG9BV4cBPzaFbSYLSQqYdr6aPUmC" height="200"/>

### (6) man, {a}, {b}
<img src="https://drive.google.com/uc?export=view&id=1z9rLeG9BV4cBPzaFbSYLSQqYdr6aPUmC" height="200"/>

### (7) man, music, {a}, {b}
<img src="https://drive.google.com/uc?export=view&id=1z9rLeG9BV4cBPzaFbSYLSQqYdr6aPUmC" height="200"/>

---

## 4. Diffusion Model Deployment
- Model: **small-stable-diffusion-v0**  
- Deployed on: **NVIDIA Jetson Xavier**
