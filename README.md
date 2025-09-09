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

A musician whose music style is {a} and {b}.
![圖片說明](https://drive.google.com/uc?export=view&id=1d4E4_otNC7CNWVTJdfxTrlZOYOskKc9I)


A musician who is {a} and {b}.
A man who is {a} and {b}.
musician, {a}, {b}
man, {a}, {b}
man, music, {a}, {b}

---

## 4. Diffusion Model Deployment
- Model: **small-stable-diffusion-v0**  
- Deployed on: **NVIDIA Jetson Xavier**
