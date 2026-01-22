import tkinter as tk
import tkinter.scrolledtext as st
import os
import re
from PIL import Image, ImageTk

class ManualViewer:
    def __init__(self, title="Application Manual", geometry="1000x800", bg_color="#2c3e50", pages=None):
        self.root = None
        self.doc_text = None
        self.doc_images = []
        self.table_buffer = []
        self.title = title
        self.geometry = geometry
        self.bg_color = bg_color
        
        # Default Pages if None provided
        if pages is None:
            self.pages = [] # Generic: No default content
        else:
            self.pages = pages # List of tuples: (Button Text, Filename, ColorHex[Optional])

    def show(self):
        if self.root is not None and self.root.winfo_exists():
            self.root.lift()
            return
            
        # Use existing Tk root if available, else create new
        try:
            self.root = tk.Toplevel()
        except RuntimeError:
            self.root = tk.Tk() # Fallback if no root exists
            
        self.root.title(self.title)
        self.root.geometry(self.geometry)
        self.root.configure(bg=self.bg_color)
        
        # Sidebar
        sidebar_bg = "#34495e"
        
        sidebar = tk.Frame(self.root, bg=sidebar_bg, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # [NEW] Resize Handling
        self.root.bind("<Configure>", self.on_window_resize)
        
        # Content Area
        content_frame = tk.Frame(self.root, bg="white")
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Text Widget
        self.doc_text = st.ScrolledText(content_frame, wrap=tk.WORD, padx=20, pady=20, font=("Georgia", 11))
        self.doc_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure Tags
        self.doc_text.tag_config("h1", font=("Helvetica", 24, "bold"), spacing3=10, foreground="#2c3e50")
        self.doc_text.tag_config("h2", font=("Helvetica", 18, "bold"), spacing3=10, foreground="#e67e22")
        self.doc_text.tag_config("h3", font=("Helvetica", 14, "bold"), spacing3=5, foreground="#2980b9")
        self.doc_text.tag_config("h4", font=("Helvetica", 12, "bold"), spacing3=5, foreground="#8e44ad")
        self.doc_text.tag_config("bold", font=("Georgia", 11, "bold"))
        self.doc_text.tag_config("italic", font=("Georgia", 11, "italic"))
        self.doc_text.tag_config("quote", font=("Georgia", 11, "italic"), lmargin1=20, lmargin2=20, foreground="#7f8c8d")
        self.doc_text.tag_config("code", font=("Courier", 10), background="#ecf0f1")
        self.doc_text.tag_config("link", foreground="blue", underline=1, font=("Georgia", 11, "bold"))
        
        # Table Tags
        self.doc_text.tag_config("table", font="TkFixedFont", background="#ecf0f1", 
                                 tabs=("2c", "6c", "10c", "14c", "18c", "22c"))

        # Sidebar Title
        tk.Label(sidebar, text="📚 Manual", bg=sidebar_bg, fg="#bdc3c7", font=("Arial", 12, "bold")).pack(pady=20)
        
        # Dynamic Buttons
        base_style = {"fg": "white", "font": ("Arial", 11, "bold")}
        
        for page in self.pages:
            # Unpack: Label, File, [Color]
            label = page[0]
            filename = page[1]
            color = page[2] if len(page) > 2 else "#2980b9"
            
            # Use default capture issue fix (filename=filename)
            tk.Button(sidebar, text=label, bg=color, **base_style,
                      command=lambda f=filename: self.load_doc(f)).pack(fill=tk.X, padx=10, pady=5)

            
        tk.Button(sidebar, text="Close", bg="#c0392b", fg="white", command=self.root.destroy).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)
        
        # Initial Load (First page)
        if self.pages:
            self.load_doc(self.pages[0][1])

    def load_doc(self, filename):
        path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(path):
             self.doc_text.config(state=tk.NORMAL)
             self.doc_text.delete("1.0", tk.END)
             self.doc_text.insert(tk.END, f"File not found: {filename}\n(Expected at {path})")
             self.doc_text.config(state=tk.DISABLED)
             return
        
        try:
            with open(path, 'r') as f: content = f.read()
            self._render_markdown(content)
        except Exception as e:
            self.doc_text.config(state=tk.NORMAL)
            self.doc_text.delete("1.0", tk.END)
            self.doc_text.insert(tk.END, f"Error loading file: {e}")
            self.doc_text.config(state=tk.DISABLED)

    def _render_markdown(self, content):
        self.doc_text.config(state=tk.NORMAL)
        self.doc_text.delete("1.0", tk.END)
        self.doc_images = []
        self.table_buffer = []

        def flush_table_buffer():
            if not self.table_buffer: return
            
            # Create a Frame for the table
            table_frame = tk.Frame(self.doc_text, bg="#bdc3c7", padx=1, pady=1) # Border effect
            
            # Parse Data
            raw_rows = []
            for l in self.table_buffer:
                # | Cell | Cell | -> ['Cell', 'Cell']
                cells = [c.strip() for c in l.strip().split('|')]
                if len(cells) > 0 and cells[0] == '': cells.pop(0)
                if len(cells) > 0 and cells[-1] == '': cells.pop(-1)
                raw_rows.append(cells)
            
            # Render
            for r_idx, row_data in enumerate(raw_rows):
                # Check for Divider Row (e.g. ---)
                if all(c.replace('-', '').replace(':', '').strip() == '' for c in row_data):
                    continue
                    
                is_header = (r_idx == 0)
                bg_color = "#34495e" if is_header else ("#ecf0f1" if r_idx % 2 == 0 else "#ffffff")
                fg_color = "white" if is_header else "black"
                font_style = ("Arial", 11, "bold") if is_header else ("Arial", 10)
                
                for c_idx, cell_text in enumerate(row_data):
                    clean_text = cell_text.replace("**", "")
                    lbl = tk.Label(table_frame, text=clean_text, font=font_style, 
                                   bg=bg_color, fg=fg_color, padx=10, pady=5, borderwidth=1, relief="solid")
                    lbl.grid(row=r_idx, column=c_idx, sticky="nsew")
            
            # Embed the frame
            self.doc_text.window_create(tk.END, window=table_frame)
            self.doc_text.insert(tk.END, "\n")
            self.table_buffer = []

        def insert_with_tags(text, context_tag=None):
            # [NEW] Helper to merge tags
            def get_tags(extra=None):
                tags = []
                if context_tag: tags.append(context_tag)
                if extra: tags.append(extra)
                return tuple(tags) if tags else None

            # [NEW] Hyperlink Parsing: [Text](Target)
            link_pattern = r'(\[[^\]]+\]\([^)]+\))'
            parts = re.split(link_pattern, text)
            
            for p in parts:
                match = re.match(r'\[([^\]]+)\]\(([^)]+)\)', p)
                if match:
                    link_text = match.group(1)
                    target = match.group(2)
                    
                    # Insert Link with Tag
                    tag_name = f"link_{len(self.doc_text.tag_names())}"
                    # Combine link tag with context tag (e.g. h1+link)
                    combined_tags = list(get_tags("link") or [])
                    combined_tags.append(tag_name)
                    
                    self.doc_text.insert(tk.END, link_text, tuple(combined_tags))
                    
                    # Bind Click
                    def on_link_click(e, t=target):
                        # print(f"Link Clicked: {t}")
                        self.load_doc(t)
                        
                    self.doc_text.tag_bind(tag_name, "<Button-1>", on_link_click)
                else:
                    # Proceed with Bold/Italic Parsing on non-link text
                    parse_formatting(p, get_tags)

        def parse_formatting(text, tag_getter):
            # 1. Bold Parsing (**text**)
            parts = re.split(r'(\*\*.*?\*\*)', text)
            for p in parts:
                if p.startswith('**') and p.endswith('**') and len(p) > 4:
                    self.doc_text.insert(tk.END, p[2:-2], tag_getter("bold"))
                else:
                    # 2. Italic Parsing (*text*)
                    sub_parts = re.split(r'(\*.*?\*)', p)
                    for sp in sub_parts:
                        if sp.startswith('*') and sp.endswith('*') and len(sp) > 2:
                            self.doc_text.insert(tk.END, sp[1:-1], tag_getter("italic"))
                        else:
                            self.doc_text.insert(tk.END, sp, tag_getter())

        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()

            # Check for Table Line
            if stripped.startswith("|"):
                self.table_buffer.append(stripped)
                continue
            else:
                flush_table_buffer()
            
            # 1. Headers
            if stripped.startswith('# '):
                insert_with_tags(stripped[2:], "h1")
                self.doc_text.insert(tk.END, "\n")
            elif stripped.startswith('## '):
                insert_with_tags(stripped[3:], "h2")
                self.doc_text.insert(tk.END, "\n")
            elif stripped.startswith('### '):
                insert_with_tags(stripped[4:], "h3")
                self.doc_text.insert(tk.END, "\n")
            elif stripped.startswith('#### '):
                insert_with_tags(stripped[5:], "h4")
                self.doc_text.insert(tk.END, "\n")
            
            # 2. Blockquotes
            elif stripped.startswith('> '):
                 insert_with_tags(stripped[2:], "quote")
                 self.doc_text.insert(tk.END, "\n")

            # [NEW] Horizontal Rule
            elif stripped.startswith('---') or stripped.startswith('***'):
                # Draw a line using a Frame with explicit height and background color
                # We need a small font to prevent large gaps, or just use window_create
                hr_frame = tk.Frame(self.doc_text, height=2, bg="#bdc3c7", width=600) # Width is somewhat arbitrary, effectively fill X?
                # Text widget width logic is tricky, but fixed width is OK for now.
                # Better: Use a canvas line? Or simple text.
                # A Frame with 'stretch' behavior in Text widget is hard.
                # Let's try a full-width line using unicode if frame fails, but Frame is standard for this.
                # Width=800 should cover most of the view.
                hr_frame = tk.Frame(self.doc_text, height=2, bg="#bdc3c7", width=800)
                self.doc_text.window_create(tk.END, window=hr_frame)
                self.doc_text.insert(tk.END, "\n")

            # 3. Images: ![alt](path)
            elif stripped.startswith("!["):
                try:
                    s = line.find("(") + 1
                    e = line.find(")")
                    path = line[s:e]
                    full_path = os.path.join(os.getcwd(), path)
                    
                    if os.path.exists(full_path):
                        img = Image.open(full_path)
                        # Resize to fit
                        base_width = 500
                        w_percent = (base_width / float(img.size[0]))
                        h_size = int((float(img.size[1]) * float(w_percent)))
                        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
                        
                        photo = ImageTk.PhotoImage(img)
                        self.doc_images.append(photo)
                        self.doc_text.image_create(tk.END, image=photo)
                        self.doc_text.insert(tk.END, "\n")
                    else:
                         self.doc_text.insert(tk.END, f"[Image not found: {path}]\n")
                except Exception as e:
                    self.doc_text.insert(tk.END, f"[Image Load Failed: {e}]\n")
                 
            # 4. List Items
            elif stripped.startswith('* ') or stripped.startswith('- '):
                 self.doc_text.insert(tk.END, "  • ")
                 insert_with_tags(stripped[2:])
                 self.doc_text.insert(tk.END, "\n") # [FIX] Add newline for list items
            
            # 5. Normal Text
            else:
                insert_with_tags(stripped)
                self.doc_text.insert(tk.END, "\n") # Restore newline if was stripped

        flush_table_buffer() 
        self.doc_text.config(state=tk.DISABLED)
    
    def on_window_resize(self, event):
        if event.widget == self.root:
            self.root.update()
