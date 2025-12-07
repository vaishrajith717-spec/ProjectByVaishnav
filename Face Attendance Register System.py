import cv2
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime


# ======================== ATTENDANCE SYSTEM ========================
class FaceAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.known_faces = {}
        self.data_file = "attendance_face_data.pkl"
        self.attendance_file = "attendance_records.txt"
        self.active_sessions = {}

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_trained = False
        self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data['faces']
                if self.known_faces:
                    self.train_recognizer()
            except:
                pass

    def save_known_faces(self):
        data = {'faces': self.known_faces}
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)

    def train_recognizer(self):
        if not self.known_faces:
            return
        faces, labels, label_map = [], [], {}
        for idx, (username, face_data) in enumerate(self.known_faces.items()):
            faces.extend(face_data['images'])
            labels.extend([idx] * len(face_data['images']))
            label_map[idx] = username
        if faces:
            self.recognizer.train(faces, np.array(labels))
            self.label_map = label_map
            self.face_trained = True

    def register_new_user(self, frame, username):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) == 0:
            return False, "No face detected"
        if len(faces) > 1:
            return False, "Multiple faces"
        (x, y, w, h) = faces[0]
        face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        if username in self.known_faces:
            self.known_faces[username]['images'].append(face_roi)
        else:
            self.known_faces[username] = {'images': [face_roi]}
        self.train_recognizer()
        self.save_known_faces()
        return True, "Registered!"

    def recognize_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        recognized_faces = []
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            name, confidence = "Unknown", 0
            if self.face_trained:
                label, conf = self.recognizer.predict(face_roi)
                if conf < 70:
                    name = self.label_map.get(label, "Unknown")
                    confidence = 100 - conf
            recognized_faces.append({'location': (x, y, w, h), 'name': name, 'confidence': confidence})
        return recognized_faces

    def time_in(self, name):
        if name in self.active_sessions:
            return False, f"{name} already checked in!"
        time_in = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.active_sessions[name] = {'time_in': time_in}
        with open(self.attendance_file, 'a') as f:
            f.write(f"{name},TIME IN,{time_in}\n")
        return True, f"✅ {name} - Time In: {time_in.split()[1]}"

    def time_out(self, name):
        if name not in self.active_sessions:
            return False, f"{name} not checked in!"
        time_out = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_in = self.active_sessions[name]['time_in']
        time_in_dt = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S")
        time_out_dt = datetime.strptime(time_out, "%Y-%m-%d %H:%M:%S")
        duration = time_out_dt - time_in_dt
        with open(self.attendance_file, 'a') as f:
            f.write(f"{name},TIME OUT,{time_out},Duration: {duration}\n")
        del self.active_sessions[name]
        return True, f"✅ {name} - Time Out: {time_out.split()[1]} ({duration})"

    def show_register_book(self):
        """Display attendance register book in a GUI window"""
        register_window = tk.Toplevel()
        register_window.title("📖 Attendance Register Book")
        register_window.geometry("900x600")
        register_window.configure(bg='#ECF0F1')

        # Header
        header = tk.Frame(register_window, bg='#2C3E50', height=80)
        header.pack(fill='x')
        tk.Label(header, text="📖 Attendance Register Book",
                 font=('Arial', 24, 'bold'), bg='#2C3E50', fg='white').pack(pady=20)

        # Content frame with scrollbar
        content = tk.Frame(register_window, bg='#ECF0F1')
        content.pack(expand=True, fill='both', padx=20, pady=20)

        # Create canvas and scrollbar
        canvas = tk.Canvas(content, bg='white')
        scrollbar = tk.Scrollbar(content, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Table header
        header_frame = tk.Frame(scrollable_frame, bg='#34495E', relief='raised', borderwidth=2)
        header_frame.pack(fill='x', pady=(0, 10))

        tk.Label(header_frame, text="Name", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white', width=25).grid(row=0, column=0, padx=5, pady=10)
        tk.Label(header_frame, text="Time In", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white', width=20).grid(row=0, column=1, padx=5, pady=10)
        tk.Label(header_frame, text="Time Out", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white', width=20).grid(row=0, column=2, padx=5, pady=10)
        tk.Label(header_frame, text="Duration", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white', width=15).grid(row=0, column=3, padx=5, pady=10)

        # Read attendance records with improved parsing
        today_records = {}
        today = datetime.now().strftime("%Y-%m-%d")

        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = [p.strip() for p in line.split(',')]

                    if len(parts) >= 3:
                        name = parts[0]
                        action = parts[1].upper()
                        timestamp_str = parts[2]

                        # Parse timestamp
                        if ' ' in timestamp_str:
                            date_part, time_part = timestamp_str.split(' ', 1)
                        else:
                            date_part = today
                            time_part = timestamp_str

                        # Only process today's records
                        if date_part != today:
                            continue

                        # Initialize record if needed
                        if name not in today_records:
                            today_records[name] = {'time_in': None, 'time_out': None, 'duration': None}

                        # Check action type
                        if "LOGIN" in action or "TIME IN" in action or action == "IN":
                            today_records[name]['time_in'] = time_part
                        elif "LOGOUT" in action or "TIME OUT" in action or action == "OUT":
                            today_records[name]['time_out'] = time_part
                            # Extract duration if present
                            if len(parts) >= 4:
                                for part in parts[3:]:
                                    if "Duration" in part:
                                        duration_str = part.split("Duration:")[-1].strip()
                                        today_records[name]['duration'] = duration_str
                                        break

        # Display records
        row_num = 0
        for name, record in today_records.items():
            row_frame = tk.Frame(scrollable_frame, bg='white', relief='solid', borderwidth=1)
            row_frame.pack(fill='x', pady=2)

            bg_color = '#E8F8F5' if row_num % 2 == 0 else 'white'

            tk.Label(row_frame, text=name, font=('Arial', 11),
                     bg=bg_color, anchor='w', width=25).grid(row=0, column=0, padx=5, pady=8, sticky='w')

            time_in = record['time_in'] if record['time_in'] else '-'
            time_in_color = '#27AE60' if record['time_in'] else '#E74C3C'
            tk.Label(row_frame, text=time_in, font=('Arial', 11),
                     bg=bg_color, fg=time_in_color, width=20).grid(row=0, column=1, padx=5, pady=8)

            time_out = record['time_out'] if record['time_out'] else 'Still In'
            time_out_color = '#E67E22' if record['time_out'] else '#3498DB'
            tk.Label(row_frame, text=time_out, font=('Arial', 11),
                     bg=bg_color, fg=time_out_color, width=20).grid(row=0, column=2, padx=5, pady=8)

            duration = record['duration'] if record['duration'] else '-'
            tk.Label(row_frame, text=duration, font=('Arial', 11),
                     bg=bg_color, width=15).grid(row=0, column=3, padx=5, pady=8)

            row_num += 1

        # Show active sessions
        if self.active_sessions:
            for name in self.active_sessions:
                if name not in today_records:
                    row_frame = tk.Frame(scrollable_frame, bg='white', relief='solid', borderwidth=1)
                    row_frame.pack(fill='x', pady=2)

                    bg_color = '#FEF9E7'

                    tk.Label(row_frame, text=name, font=('Arial', 11, 'bold'),
                             bg=bg_color, anchor='w', width=25).grid(row=0, column=0, padx=5, pady=8, sticky='w')
                    tk.Label(row_frame, text=self.active_sessions[name]['time_in'].split()[1],
                             font=('Arial', 11), bg=bg_color, fg='#27AE60', width=20).grid(row=0, column=1, padx=5,
                                                                                           pady=8)
                    tk.Label(row_frame, text="Still In", font=('Arial', 11, 'bold'),
                             bg=bg_color, fg='#3498DB', width=20).grid(row=0, column=2, padx=5, pady=8)
                    tk.Label(row_frame, text="-", font=('Arial', 11),
                             bg=bg_color, width=15).grid(row=0, column=3, padx=5, pady=8)

        if not today_records and not self.active_sessions:
            tk.Label(scrollable_frame, text="No attendance records for today",
                     font=('Arial', 14), bg='white', fg='#7F8C8D').pack(pady=50)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Close button
        tk.Button(register_window, text="Close", font=('Arial', 12, 'bold'),
                  bg='#E74C3C', fg='white', width=15, height=2,
                  command=register_window.destroy).pack(pady=10)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("ERROR: Camera not found!")
            return

        message, msg_time = "", 0
        print("\n" + "=" * 70)
        print("ATTENDANCE SYSTEM - AUTO TIME IN/OUT")
        print("=" * 70)
        print("Press 'v' to View Register Book | 'q' to Quit")
        print("=" * 70)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            now = datetime.now().timestamp()

            faces = self.recognize_face(frame)
            for face in faces:
                x, y, w, h = face['location']
                name = face['name']

                # Auto Time IN/OUT
                if name != "Unknown":
                    if name not in self.active_sessions:
                        # Auto Time IN
                        success, msg = self.time_in(name)
                        if success:
                            message, msg_time = msg, now

                checked = name in self.active_sessions
                color = (255, 165, 0) if checked else ((0, 255, 0) if name != "Unknown" else (0, 0, 255))
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                label = f"{name}"
                if name != "Unknown":
                    label += f" ({face['confidence']:.1f}%)"
                    if checked:
                        label += " [CHECKED IN]"
                cv2.rectangle(display, (x, y - 35), (x + w, y), color, cv2.FILLED)
                cv2.putText(display, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(display, "ATTENDANCE SYSTEM - Press 'v' for Register Book", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show active sessions
            y_pos = 60
            if self.active_sessions:
                cv2.putText(display, "Currently Checked In:", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                for user, session in list(self.active_sessions.items())[:5]:
                    y_pos += 25
                    cv2.putText(display, f"  {user}: {session['time_in'].split()[1]}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if now - msg_time < 3 and message:
                cv2.putText(display, message, (10, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Attendance System', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Auto Time OUT for all active sessions
                for name in list(self.active_sessions.keys()):
                    self.time_out(name)
                break
            elif key == ord('v'):
                # Show register book
                self.show_register_book()

        cap.release()
        cv2.destroyAllWindows()

        print("\n\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print("\nAttendance records saved to:", self.attendance_file)
        print("=" * 70)


# ======================== LOGIN SYSTEM ========================
class FaceLoginSystem:
    def __init__(self, parent_root):
        self.parent_root = parent_root
        self.known_faces = {}
        self.data_file = "users_face_data.pkl"
        self.login_log = "login_logs.txt"
        self.attendance_file = "attendance_records.txt"  # Share with Register Book
        self.current_session = None

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_trained = False

        # Load users before creating window
        self.load_users()

        # Create window
        self.root = tk.Toplevel(parent_root)
        self.root.title("Face Login System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2C3E50')

        # Make sure window appears on top
        self.root.lift()
        self.root.focus_force()

        self.cap = None
        self.create_login_screen()

    def load_users(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                if self.known_faces:
                    self.train_recognizer()
            except:
                pass

    def save_users(self):
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def train_recognizer(self):
        if not self.known_faces:
            return
        faces, labels, self.label_map = [], [], {}
        for idx, (username, data) in enumerate(self.known_faces.items()):
            faces.extend(data['face_samples'])
            labels.extend([idx] * len(data['face_samples']))
            self.label_map[idx] = username
        if faces:
            self.recognizer.train(faces, np.array(labels))
            self.face_trained = True

    def create_login_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        # Make sure window is visible
        self.root.deiconify()
        self.root.lift()

        header = tk.Frame(self.root, bg='#34495E', height=80)
        header.pack(fill='x')
        tk.Label(header, text="🔐 Face Login System", font=('Arial', 24, 'bold'),
                 bg='#34495E', fg='white').pack(pady=20)

        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(expand=True, fill='both', padx=50, pady=30)

        tk.Label(main_frame, text="Welcome! Choose an option:", font=('Arial', 16),
                 bg='#2C3E50', fg='white').pack(pady=20)

        self.video_frame = tk.Label(main_frame, bg='#34495E', width=640, height=480)
        self.video_frame.pack(pady=20)

        button_frame = tk.Frame(main_frame, bg='#2C3E50')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="🔑 Login", font=('Arial', 14, 'bold'), bg='#27AE60',
                  fg='white', width=20, height=2, command=self.start_login).grid(row=0, column=0, padx=10)
        tk.Button(button_frame, text="📝 Register", font=('Arial', 14, 'bold'), bg='#3498DB',
                  fg='white', width=20, height=2, command=self.start_registration).grid(row=0, column=1, padx=10)

        self.status_label = tk.Label(main_frame, text="", font=('Arial', 12), bg='#2C3E50', fg='#ECF0F1')
        self.status_label.pack(pady=10)

        print("Starting webcam preview...")
        self.start_webcam_preview()

    def start_webcam_preview(self):
        if self.cap is None or not self.cap.isOpened():
            for i in range(3):
                self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    print(f"Login System: Camera opened on index {i}")
                    break

            if not self.cap.isOpened():
                messagebox.showerror("Camera Error",
                                     "Could not access camera.\n\n"
                                     "Please check:\n"
                                     "1. Camera permissions\n"
                                     "2. Camera not used by other apps")
                return

        self.update_preview()

    def update_preview(self):
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (640, 480))
                    img = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.video_frame.imgtk = img
                    self.video_frame.configure(image=img)

            if self.root.winfo_exists():
                self.root.after(10, self.update_preview)
        except Exception as e:
            print(f"Preview error: {e}")
            if self.root.winfo_exists():
                self.root.after(100, self.update_preview)

    def start_login(self):
        self.status_label.config(text="🔍 Scanning...", fg='#F39C12')
        self.root.update()
        if not self.known_faces:
            messagebox.showwarning("No Users", "Register first")
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) == 0:
            self.status_label.config(text="❌ No face", fg='#E74C3C')
            return
        (x, y, w, h) = faces[0]
        face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        if self.face_trained:
            label, confidence = self.recognizer.predict(face_roi)
            if confidence < 70:
                username = self.label_map.get(label, "Unknown")
                time_in = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.current_session = {'username': username, 'time_in': time_in}
                with open(self.login_log, 'a') as f:
                    f.write(f"{username},LOGIN,{time_in}\n")
                self.status_label.config(text=f"✅ Welcome {username}", fg='#27AE60')
                self.root.after(1000, lambda: self.show_dashboard(username, time_in))
            else:
                self.status_label.config(text="❌ Not recognized", fg='#E74C3C')

    def start_registration(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Register")
        dialog.geometry("400x250")
        dialog.configure(bg='#34495E')
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(dialog, text="Username:", font=('Arial', 14), bg='#34495E', fg='white').pack(pady=20)
        username_var = tk.StringVar()
        tk.Entry(dialog, textvariable=username_var, font=('Arial', 12), width=30).pack(pady=10)

        def register():
            username = username_var.get().strip()
            if not username:
                messagebox.showwarning("Invalid", "Enter username")
                return
            if username in self.known_faces:
                messagebox.showwarning("Exists", "Username exists")
                return
            dialog.destroy()
            self.capture_face_samples(username)

        tk.Button(dialog, text="Start", command=register, font=('Arial', 12, 'bold'),
                  bg='#27AE60', fg='white', width=20, height=2).pack(pady=20)

    def capture_face_samples(self, username):
        samples, required = [], 5

        def capture():
            if len(samples) >= required:
                self.known_faces[username] = {
                    'face_samples': samples,
                    'registered_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.train_recognizer()
                self.save_users()
                self.status_label.config(text=f"✅ {username} registered!", fg='#27AE60')
                messagebox.showinfo("Success", f"{username} registered!")
                return
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    samples.append(cv2.resize(gray[y:y + h, x:x + w], (200, 200)))
                    self.status_label.config(text=f"✅ Sample {len(samples)}/{required}", fg='#27AE60')
                    self.root.after(1000, capture)
                else:
                    self.status_label.config(text=f"⚠️ One face {len(samples)}/{required}", fg='#F39C12')
                    self.root.after(500, capture)

        capture()

    def show_dashboard(self, username, time_in):
        for widget in self.root.winfo_children():
            widget.destroy()
        if self.cap:
            self.cap.release()

        header = tk.Frame(self.root, bg='#27AE60', height=100)
        header.pack(fill='x')
        tk.Label(header, text=f"Welcome, {username}! 👋", font=('Arial', 28, 'bold'),
                 bg='#27AE60', fg='white').pack(pady=30)

        content = tk.Frame(self.root, bg='#ECF0F1')
        content.pack(expand=True, fill='both')

        info_frame = tk.Frame(content, bg='white', relief='raised', borderwidth=2)
        info_frame.pack(pady=50, padx=50, fill='both', expand=True)

        tk.Label(info_frame, text="✅ Login Successful!", font=('Arial', 24, 'bold'),
                 bg='white', fg='#27AE60').pack(pady=30)

        user_data = self.known_faces.get(username, {})

        # Create details frame with better layout
        details_frame = tk.Frame(info_frame, bg='white')
        details_frame.pack(pady=20)

        # Username
        tk.Label(details_frame, text="👤 Username:", font=('Arial', 14, 'bold'),
                 bg='white', fg='#2C3E50', anchor='w').grid(row=0, column=0, sticky='w', padx=20, pady=10)
        tk.Label(details_frame, text=username, font=('Arial', 14),
                 bg='white', fg='#7F8C8D').grid(row=0, column=1, sticky='w', padx=20, pady=10)

        # Registered Date
        tk.Label(details_frame, text="📅 Registered:", font=('Arial', 14, 'bold'),
                 bg='white', fg='#2C3E50', anchor='w').grid(row=1, column=0, sticky='w', padx=20, pady=10)
        tk.Label(details_frame, text=user_data.get('registered_date', 'N/A'),
                 font=('Arial', 14), bg='white', fg='#7F8C8D').grid(row=1, column=1, sticky='w', padx=20, pady=10)

        # Time In
        tk.Label(details_frame, text="🕐 Time In:", font=('Arial', 14, 'bold'),
                 bg='white', fg='#2C3E50', anchor='w').grid(row=2, column=0, sticky='w', padx=20, pady=10)
        tk.Label(details_frame, text=time_in, font=('Arial', 14),
                 bg='white', fg='#27AE60').grid(row=2, column=1, sticky='w', padx=20, pady=10)

        # Time Out (placeholder)
        tk.Label(details_frame, text="🕐 Time Out:", font=('Arial', 14, 'bold'),
                 bg='white', fg='#2C3E50', anchor='w').grid(row=3, column=0, sticky='w', padx=20, pady=10)
        self.time_out_label = tk.Label(details_frame, text="Active Session",
                                       font=('Arial', 14), bg='white', fg='#E67E22')
        self.time_out_label.grid(row=3, column=1, sticky='w', padx=20, pady=10)

        # Status
        tk.Label(details_frame, text="✨ Status:", font=('Arial', 14, 'bold'),
                 bg='white', fg='#2C3E50', anchor='w').grid(row=4, column=0, sticky='w', padx=20, pady=10)
        tk.Label(details_frame, text="Authenticated", font=('Arial', 14),
                 bg='white', fg='#27AE60').grid(row=4, column=1, sticky='w', padx=20, pady=10)

        # Session duration timer
        self.time_label = tk.Label(info_frame, text="", font=('Arial', 16, 'bold'),
                                   bg='white', fg='#3498DB')
        self.time_label.pack(pady=20)
        self.update_session_time(time_in)

        tk.Button(info_frame, text="🚪 Logout", command=lambda: self.logout(username, time_in),
                  font=('Arial', 14, 'bold'), bg='#E74C3C', fg='white',
                  width=20, height=2).pack(pady=30)

    def update_session_time(self, time_in):
        if not hasattr(self, 'time_label') or not self.time_label.winfo_exists():
            return
        time_in_dt = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S")
        duration = datetime.now() - time_in_dt
        h, r = divmod(int(duration.total_seconds()), 3600)
        m, s = divmod(r, 60)
        self.time_label.config(text=f"⏱️ Duration: {h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, lambda: self.update_session_time(time_in))

    def logout(self, username, time_in):
        time_out = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            if self.current_session:
                time_in_dt = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S")
                time_out_dt = datetime.strptime(time_out, "%Y-%m-%d %H:%M:%S")
                duration = time_out_dt - time_in_dt

                # Log to login file
                with open(self.login_log, 'a') as f:
                    f.write(f"{username},LOGOUT,{time_out},Duration: {duration}\n")

                # Also save to attendance file for Register Book
                with open(self.attendance_file, 'a') as f:
                    f.write(f"{username},TIME OUT,{time_out},Duration: {duration}\n")

                # Format time for display
                time_in_12hr = time_in_dt.strftime("%I:%M %p")
                time_out_12hr = time_out_dt.strftime("%I:%M %p")

                messagebox.showinfo("Logged Out",
                                    f"Session Summary\n"
                                    f"{'=' * 40}\n\n"
                                    f"Name: {username}\n"
                                    f"Time In: {time_in_12hr}\n"
                                    f"Time Out: {time_out_12hr}\n"
                                    f"Duration: {duration}\n\n"
                                    f"Goodbye {username}! 👋")
        except Exception as e:
            print(f"Logout error: {e}")
            messagebox.showinfo("Logged Out", f"Goodbye {username}!")

        self.create_login_screen()


# ======================== REGISTER BOOK VIEWER ========================
class AttendanceRegisterViewer:
    def __init__(self, parent_root):
        self.parent_root = parent_root
        self.attendance_file = "attendance_records.txt"

        self.root = tk.Toplevel(parent_root)
        self.root.title("📖 Attendance Register Book")
        self.root.geometry("1100x700")
        self.root.configure(bg='#ECF0F1')

        self.create_ui()
        self.load_attendance()

    def create_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#2C3E50', height=100)
        header.pack(fill='x')

        tk.Label(header, text="📖 Attendance Register Book",
                 font=('Arial', 28, 'bold'), bg='#2C3E50', fg='white').pack(pady=20)

        date_label = tk.Label(header, text=f"Date: {datetime.now().strftime('%B %d, %Y')}",
                              font=('Arial', 14), bg='#2C3E50', fg='#ECF0F1')
        date_label.pack(pady=5)

        # Toolbar
        toolbar = tk.Frame(self.root, bg='#34495E', height=60)
        toolbar.pack(fill='x')

        button_frame = tk.Frame(toolbar, bg='#34495E')
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="🔄 Refresh", font=('Arial', 11, 'bold'),
                  bg='#3498DB', fg='white', width=12, height=1,
                  command=self.load_attendance).grid(row=0, column=0, padx=5)

        tk.Button(button_frame, text="📊 Export", font=('Arial', 11, 'bold'),
                  bg='#27AE60', fg='white', width=12, height=1,
                  command=self.export_to_file).grid(row=0, column=1, padx=5)

        # Stats frame
        stats_frame = tk.Frame(self.root, bg='white', height=80)
        stats_frame.pack(fill='x', padx=20, pady=10)

        self.total_label = tk.Label(stats_frame, text="Total Present: 0",
                                    font=('Arial', 12, 'bold'), bg='white', fg='#27AE60')
        self.total_label.pack(side='left', padx=30, pady=10)

        self.active_label = tk.Label(stats_frame, text="Currently In: 0",
                                     font=('Arial', 12, 'bold'), bg='white', fg='#3498DB')
        self.active_label.pack(side='left', padx=30, pady=10)

        self.completed_label = tk.Label(stats_frame, text="Checked Out: 0",
                                        font=('Arial', 12, 'bold'), bg='white', fg='#E67E22')
        self.completed_label.pack(side='left', padx=30, pady=10)

        # Main content frame with scrollbar
        content = tk.Frame(self.root, bg='#ECF0F1')
        content.pack(expand=True, fill='both', padx=20, pady=10)

        # Create canvas and scrollbar
        canvas = tk.Canvas(content, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(content, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='white')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Auto-refresh every 5 seconds
        self.auto_refresh()

    def load_attendance(self):
        """Load attendance records and display them"""
        # Clear existing rows
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Table header
        header_frame = tk.Frame(self.scrollable_frame, bg='#34495E', relief='raised', borderwidth=2)
        header_frame.pack(fill='x', pady=(0, 5))

        # Configure grid column weights for better alignment
        header_frame.grid_columnconfigure(0, weight=1, minsize=80)
        header_frame.grid_columnconfigure(1, weight=3, minsize=250)
        header_frame.grid_columnconfigure(2, weight=2, minsize=150)
        header_frame.grid_columnconfigure(3, weight=2, minsize=150)
        header_frame.grid_columnconfigure(4, weight=2, minsize=120)
        header_frame.grid_columnconfigure(5, weight=2, minsize=120)

        tk.Label(header_frame, text="Sr.No", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white').grid(row=0, column=0, padx=10, pady=12, sticky='nsew')
        tk.Label(header_frame, text="Name", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white').grid(row=0, column=1, padx=10, pady=12, sticky='w')
        tk.Label(header_frame, text="Time In", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white').grid(row=0, column=2, padx=10, pady=12, sticky='nsew')
        tk.Label(header_frame, text="Time Out", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white').grid(row=0, column=3, padx=10, pady=12, sticky='nsew')
        tk.Label(header_frame, text="Duration", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white').grid(row=0, column=4, padx=10, pady=12, sticky='nsew')
        tk.Label(header_frame, text="Status", font=('Arial', 12, 'bold'),
                 bg='#34495E', fg='white').grid(row=0, column=5, padx=10, pady=12, sticky='nsew')

        # Read attendance records
        today_records = {}
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        name, action, timestamp = parts[0], parts[1], parts[2]
                        date = timestamp.split()[0]
                        today = datetime.now().strftime("%Y-%m-%d")

                        if date == today:
                            if name not in today_records:
                                today_records[name] = {'time_in': None, 'time_out': None, 'duration': None}

                            if action == "TIME IN":
                                today_records[name]['time_in'] = timestamp.split()[1]
                            elif action == "TIME OUT" and len(parts) >= 4:
                                today_records[name]['time_out'] = timestamp.split()[1]
                                if "Duration:" in parts[3]:
                                    today_records[name]['duration'] = parts[3].split("Duration:")[1].strip()

        # Update stats
        total_present = len(today_records)
        still_in = sum(1 for r in today_records.values() if r['time_out'] is None)
        checked_out = total_present - still_in

        self.total_label.config(text=f"Total Present: {total_present}")
        self.active_label.config(text=f"Currently In: {still_in}")
        self.completed_label.config(text=f"Checked Out: {checked_out}")

        # Display records
        if today_records:
            sr_no = 1
            for name, record in today_records.items():
                row_frame = tk.Frame(self.scrollable_frame, bg='white', relief='solid', borderwidth=1)
                row_frame.pack(fill='x', pady=2)

                # Configure grid column weights to match header
                row_frame.grid_columnconfigure(0, weight=1, minsize=80)
                row_frame.grid_columnconfigure(1, weight=3, minsize=250)
                row_frame.grid_columnconfigure(2, weight=2, minsize=150)
                row_frame.grid_columnconfigure(3, weight=2, minsize=150)
                row_frame.grid_columnconfigure(4, weight=2, minsize=120)
                row_frame.grid_columnconfigure(5, weight=2, minsize=120)

                bg_color = '#E8F8F5' if sr_no % 2 == 0 else 'white'

                # Sr No
                tk.Label(row_frame, text=str(sr_no), font=('Arial', 11),
                         bg=bg_color).grid(row=0, column=0, padx=10, pady=12, sticky='nsew')

                # Name
                tk.Label(row_frame, text=name, font=('Arial', 11, 'bold'),
                         bg=bg_color, anchor='w').grid(row=0, column=1, padx=10, pady=12, sticky='w')

                # Time In (12-hour format)
                if record['time_in']:
                    try:
                        time_obj = datetime.strptime(record['time_in'], "%H:%M:%S")
                        time_in = time_obj.strftime("%I:%M:%S %p")
                        time_in_color = '#27AE60'
                    except:
                        time_in = record['time_in']
                        time_in_color = '#27AE60'
                else:
                    time_in = '-'
                    time_in_color = '#E74C3C'

                tk.Label(row_frame, text=time_in, font=('Arial', 11),
                         bg=bg_color, fg=time_in_color).grid(row=0, column=2, padx=10, pady=12, sticky='nsew')

                # Time Out (12-hour format)
                if record['time_out']:
                    try:
                        time_obj = datetime.strptime(record['time_out'], "%H:%M:%S")
                        time_out = time_obj.strftime("%I:%M:%S %p")
                        time_out_color = '#E67E22'
                    except:
                        time_out = record['time_out']
                        time_out_color = '#E67E22'
                else:
                    time_out = 'Still In'
                    time_out_color = '#3498DB'

                tk.Label(row_frame, text=time_out, font=('Arial', 11),
                         bg=bg_color, fg=time_out_color).grid(row=0, column=3, padx=10, pady=12, sticky='nsew')

                # Duration
                duration = record['duration'] if record['duration'] else '-'
                tk.Label(row_frame, text=duration, font=('Arial', 11),
                         bg=bg_color).grid(row=0, column=4, padx=10, pady=12, sticky='nsew')

                # Status
                status = "✓ Complete" if record['time_out'] else "● Active"
                status_color = '#27AE60' if record['time_out'] else '#F39C12'
                tk.Label(row_frame, text=status, font=('Arial', 11, 'bold'),
                         bg=bg_color, fg=status_color).grid(row=0, column=5, padx=10, pady=12, sticky='nsew')

                sr_no += 1
        else:
            tk.Label(self.scrollable_frame, text="📋 No attendance records for today",
                     font=('Arial', 16, 'bold'), bg='white', fg='#7F8C8D').pack(pady=100)

    def export_to_file(self):
        """Export attendance to a text file"""
        try:
            export_file = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(export_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("ATTENDANCE REGISTER BOOK\n")
                f.write(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"{'Sr.No':<8} {'Name':<25} {'Time In':<15} {'Time Out':<15} {'Duration':<15}\n")
                f.write("-" * 80 + "\n")

                if os.path.exists(self.attendance_file):
                    today_records = {}
                    with open(self.attendance_file, 'r') as af:
                        for line in af:
                            parts = line.strip().split(',')
                            if len(parts) >= 3:
                                name, action, timestamp = parts[0], parts[1], parts[2]
                                date = timestamp.split()[0]
                                today = datetime.now().strftime("%Y-%m-%d")

                                if date == today:
                                    if name not in today_records:
                                        today_records[name] = {'time_in': '-', 'time_out': '-', 'duration': '-'}

                                    if action == "TIME IN":
                                        today_records[name]['time_in'] = timestamp.split()[1]
                                    elif action == "TIME OUT" and len(parts) >= 4:
                                        today_records[name]['time_out'] = timestamp.split()[1]
                                        if "Duration:" in parts[3]:
                                            today_records[name]['duration'] = parts[3].split("Duration:")[1].strip()

                    sr_no = 1
                    for name, record in today_records.items():
                        f.write(
                            f"{sr_no:<8} {name:<25} {record['time_in']:<15} {record['time_out']:<15} {record['duration']:<15}\n")
                        sr_no += 1

                f.write("\n" + "=" * 80 + "\n")

            messagebox.showinfo("Export Success", f"Attendance exported to:\n{export_file}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    def auto_refresh(self):
        """Auto refresh every 5 seconds"""
        if self.root.winfo_exists():
            self.load_attendance()
            self.root.after(5000, self.auto_refresh)


# ======================== MAIN MENU ========================
class MainMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.geometry("1100x650")
        self.root.configure(bg='#1a1a2e')
        self.create_menu()

    def create_menu(self):
        header = tk.Frame(self.root, bg='#16213e', height=120)
        header.pack(fill='x')
        tk.Label(header, text="🎯 Face Recognition System", font=('Arial', 32, 'bold'),
                 bg='#16213e', fg='#eee').pack(pady=20)
        tk.Label(header, text="Choose Your System", font=('Arial', 14),
                 bg='#16213e', fg='#94a3b8').pack()

        content = tk.Frame(self.root, bg='#1a1a2e')
        content.pack(expand=True, fill='both', pady=30)

        card_frame = tk.Frame(content, bg='#1a1a2e')
        card_frame.pack()

        # Login Card
        login_card = tk.Frame(card_frame, bg='#0f3460', relief='raised', borderwidth=3)
        login_card.grid(row=0, column=0, padx=30, pady=20)
        tk.Label(login_card, text="🔐", font=('Arial', 60), bg='#0f3460').pack(pady=20)
        tk.Label(login_card, text="Login System", font=('Arial', 18, 'bold'),
                 bg='#0f3460', fg='white').pack()
        tk.Label(login_card, text="Secure login with\nattendance tracking", font=('Arial', 11),
                 bg='#0f3460', fg='#94a3b8', justify='center').pack(pady=10)
        tk.Button(login_card, text="Open Login", font=('Arial', 12, 'bold'),
                  bg='#2980b9', fg='white', width=25, height=2,
                  command=self.open_login).pack(pady=20, padx=20)

        # Register Book Viewer Card
        register_card = tk.Frame(card_frame, bg='#0f3460', relief='raised', borderwidth=3)
        register_card.grid(row=0, column=1, padx=30, pady=20)
        tk.Label(register_card, text="📖", font=('Arial', 60), bg='#0f3460').pack(pady=20)
        tk.Label(register_card, text="Register Book", font=('Arial', 18, 'bold'),
                 bg='#0f3460', fg='white').pack()
        tk.Label(register_card, text="View attendance\nrecords (No camera)", font=('Arial', 11),
                 bg='#0f3460', fg='#94a3b8', justify='center').pack(pady=10)
        tk.Button(register_card, text="View Register Book", font=('Arial', 12, 'bold'),
                  bg='#8e44ad', fg='white', width=25, height=2,
                  command=self.open_register_book).pack(pady=20, padx=20)

        footer = tk.Frame(self.root, bg='#16213e', height=60)
        footer.pack(fill='x', side='bottom')
        tk.Label(footer, text="© 2025 Face Recognition System | Powered by OpenCV",
                 font=('Arial', 10), bg='#16213e', fg='#64748b').pack(pady=20)

    def open_login(self):
        try:
            print("Opening Login System...")
            login_system = FaceLoginSystem(self.root)
            print("Login System window created")
            self.root.wait_window(login_system.root)
            print("Login System closed")
        except Exception as e:
            print(f"Error opening login system: {e}")
            messagebox.showerror("Error", f"Could not open Login System:\n{str(e)}")

    def open_register_book(self):
        register_viewer = AttendanceRegisterViewer(self.root)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MainMenu()
    app.run()


