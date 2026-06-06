# 📊 سیستم پیش‌بینی ریسک فرسودگی شغلی و تحلیل عوامل مؤثر

این پروژه یک راهکار هوشمند مبتنی بر **یادگیری ماشین (Machine Learning)** و تحلیل داده است که به سازمان‌ها و تیم‌های منابع انسانی کمک می‌کند ریسک فرسودگی شغلی کارکنان را پیش‌بینی کرده و عوامل اصلی مؤثر بر آن را شناسایی کنند.

سیستم، کارکنان را در سه سطح ریسک دسته‌بندی می‌کند:

* 🟢 ریسک پایین (Low Risk)
* 🟡 ریسک متوسط (Moderate Risk)
* 🔴 ریسک بالا (High Risk)

---

## 🎯 اهداف پروژه

### 🔍 تحلیل عوامل مؤثر بر فرسودگی شغلی

شناسایی عوامل فردی و سازمانی که بیشترین تأثیر را در ایجاد فرسودگی شغلی کارکنان دارند.

### ⚠️ تشخیص زودهنگام ریسک

شناسایی کارکنان در معرض خطر پیش از بروز مشکلات جدی، با هدف کاهش ترک خدمت و بهبود سلامت سازمانی.

### 💡 ارائه بینش‌های کاربردی

تولید تحلیل‌ها و پیشنهادهای عملی برای کمک به تصمیم‌گیری مدیران منابع انسانی و بهبود تجربه کارکنان.

---

## 📊 درباره داده‌ها

داده‌های این پروژه شامل اطلاعات **۲۰٬۰۰۰ کارمند** است که به‌صورت سنتتیک (Synthetic) و با استفاده از هوش مصنوعی تولید شده‌اند تا شرایط و الگوهای واقعی محیط‌های کاری را شبیه‌سازی کنند.

### ویژگی‌های کلیدی (Features)

#### 👤 اطلاعات فردی و شغلی

* سن
* جنسیت
* نقش شغلی
* سابقه حضور در شرکت

#### 💼 شاخص‌های مرتبط با کار

* ساعات کاری هفتگی
* تعادل کار و زندگی (Work-Life Balance)
* رضایت شغلی

#### 🧠 شاخص‌های سلامت روان و عملکرد

* سطح استرس
* میزان اضطراب
* شاخص افسردگی
* کیفیت خواب
* حمایت اجتماعی
* Presenteeism (حضور فیزیکی در محل کار همراه با کاهش بهره‌وری)

#### 🎯 متغیرهای هدف

* ریسک فرسودگی شغلی (Burnout Risk)
* ترک شرکت در سال گذشته (Left Company Last Year)

---

## 🛠️ ابزارها و فناوری‌ها

| بخش                     | فناوری‌ها                             |
| ----------------------- | ------------------------------------- |
| تحلیل و آماده‌سازی داده | Pandas, NumPy                         |
| مصورسازی داده‌ها        | Matplotlib, Seaborn                   |
| گزارش‌گیری در ترمینال   | Rich                                  |
| یادگیری ماشین           | Scikit-Learn (RandomForestClassifier) |
| ذخیره و بارگذاری مدل    | Joblib                                |
| رابط کاربری             | Streamlit                             |

---

## 🖥️ داشبورد تعاملی Streamlit

برای این پروژه یک داشبورد مدیریتی توسعه داده شده است که به مدیران منابع انسانی امکان می‌دهد:

* اطلاعات کارکنان را وارد کنند.
* ریسک فرسودگی شغلی را به‌صورت لحظه‌ای پیش‌بینی کنند.
* میزان اطمینان مدل (Confidence Score) را مشاهده کنند.
* وضعیت ریسک را با رنگ‌بندی استاندارد نمایش دهند.
* هشدارها و توصیه‌های هوشمند متناسب با شرایط هر کارمند دریافت کنند.

---

## 📄 گزارش تحلیلی و پیشنهادهای مدیریتی

 * تحلیل قدم به قدم و پیشنهادات علمی و کاربردی برای مدیران کارکنان 
---

## 🚀 راهنمای نصب و اجرای برنامه

### 1️⃣ نصب وابستگی‌ها

```bash
pip install -r requirements.txt
```

### 2️⃣ اجرای برنامه

```bash
streamlit run app.py
```

پس از اجرا، برنامه به‌صورت خودکار در مرورگر پیش‌فرض سیستم باز خواهد شد.

---

## 🎓 هدف پروژه

این پروژه با هدف یادگیری عملی و توسعه مهارت‌ها در حوزه‌های زیر طراحی و پیاده‌سازی شده است:

* تحلیل داده
* مهندسی ویژگی (Feature Engineering)
* مدل‌سازی پیش‌بینی
* تحلیل منابع انسانی (HR Analytics)
* توسعه داشبوردهای تعاملی
* استخراج بینش‌های قابل استفاده برای تصمیم‌گیری‌های سازمانی

این پروژه علاوه بر جنبه آموزشی و رزومه‌ای، نمونه‌ای از کاربرد یادگیری ماشین در بهبود سلامت سازمانی، افزایش رضایت کارکنان و پشتیبانی از تصمیم‌گیری‌های مبتنی بر داده است.
# 📊 Burnout Risk Prediction & Root Cause Analysis System

This project is an intelligent **Machine Learning** and data analytics solution designed to help organizations and HR teams predict employee burnout risk and identify the key factors contributing to it.

The system classifies employees into three burnout risk levels:

* 🟢 Low Risk
* 🟡 Moderate Risk
* 🔴 High Risk

---

## 🎯 Project Objectives

### 🔍 Burnout Root Cause Analysis

Identify both individual and organizational factors that contribute to employee burnout.

### ⚠️ Early Risk Detection

Detect employees at risk before serious issues arise, helping organizations reduce turnover and improve workforce well-being.

### 💡 Actionable Insights

Provide practical recommendations that support HR decision-making and employee retention strategies.

---

## 📊 Dataset Overview

The dataset contains information from **20,000 employees** and was synthetically generated using AI to simulate realistic workplace scenarios and employee behavior patterns.

### Key Features

#### 👤 Demographic & Employment Information

* Age
* Gender
* Job Role
* Years at Company

#### 💼 Work-Related Indicators

* Weekly Working Hours
* Work-Life Balance
* Job Satisfaction

#### 🧠 Mental Health & Performance Metrics

* Stress Level
* Anxiety Level
* Depression Score
* Sleep Quality
* Social Support
* Presenteeism

#### 🎯 Target Variables

* Burnout Risk
* Left Company Last Year

---

## 🛠️ Technologies & Tools

| Category                    | Technologies                          |
| --------------------------- | ------------------------------------- |
| Data Analysis & Preparation | Pandas, NumPy                         |
| Data Visualization          | Matplotlib, Seaborn                   |
| Terminal Reporting          | Rich                                  |
| Machine Learning            | Scikit-Learn (RandomForestClassifier) |
| Model Persistence           | Joblib                                |
| User Interface              | Streamlit                             |

---

## 🖥️ Interactive Streamlit Dashboard

A user-friendly management dashboard has been developed to enable HR professionals to:

* Enter employee information
* Predict burnout risk in real time
* View model confidence scores
* Monitor risk levels using intuitive color indicators
* Receive intelligent recommendations and alerts tailored to each employee's situation

---

## 📄 Analytical Report & Management Recommendations

* Step-by-Step Analysis and Scientifically-Backed Practical Recommendations for HR Managers (Persian)

---

## 🚀 Installation & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

After launching, the application will automatically open in your default web browser.

---

## 🎓 Project Purpose

This project was developed as a practical Data Science and Machine Learning project to strengthen skills in:

* Data Analysis
* Feature Engineering
* Predictive Modeling
* HR Analytics
* Interactive Dashboard Development
* Insight Generation for Business Decision-Making

The primary focus is educational and portfolio development while demonstrating how machine learning can support employee well-being and organizational performance.
