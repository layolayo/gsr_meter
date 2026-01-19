import tkinter as tk
import tkinter.scrolledtext as st
import os
import re

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
        if self.root is not None and tk.Toplevel.winfo_exists(str(self.root)):
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
        self.doc_text.tag_config("bold", font=("Georgia", 11, "bold"))
        self.doc_text.tag_config("italic", font=("Georgia", 11, "italic"))
        self.doc_text.tag_config("quote", font=("Georgia", 11, "italic"), lmargin1=20, lmargin2=20, foreground="#7f8c8d")
        self.doc_text.tag_config("code", font=("Courier", 10), background="#ecf0f1")
        
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

        def insert_with_tags(text):
            # 1. Bold Parsing (**text**)
            parts = re.split(r'(\*\*.*?\*\*)', text)
            for p in parts:
                if p.startswith('**') and p.endswith('**') and len(p) > 4:
                    self.doc_text.insert(tk.END, p[2:-2], "bold")
                else:
                    # 2. Italic Parsing (*text*)
                    sub_parts = re.split(r'(\*.*?\*)', p)
                    for sp in sub_parts:
                        if sp.startswith('*') and sp.endswith('*') and len(sp) > 2:
                            self.doc_text.insert(tk.END, sp[1:-1], "italic")
                        else:
                            self.doc_text.insert(tk.END, sp)
            self.doc_text.insert(tk.END, "\n")

        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            
            # 1. Headers
            if stripped.startswith('# '):
                self.doc_text.insert(tk.END, stripped[2:] + "\n", "h1")
            elif stripped.startswith('## '):
                self.doc_text.insert(tk.END, stripped[3:] + "\n", "h2")
            elif stripped.startswith('### '):
                self.doc_text.insert(tk.END, stripped[4:] + "\n", "h3")
            
            # 2. Blockquotes
            elif stripped.startswith('> '):
                 self.doc_text.insert(tk.END, stripped[2:] + "\n", "quote")
                 
            # 3. List Items
            elif stripped.startswith('* ') or stripped.startswith('- '):
                 self.doc_text.insert(tk.END, "  • ")
                 insert_with_tags(stripped[2:])
            
            # 4. Standard Text
            else:
                 # Check for table line (simple pipe detection)
                 if "|" in stripped and len(stripped) > 2:
                      self.doc_text.insert(tk.END, stripped + "\n", "code") # Temporary simple table render
                 else:
                      insert_with_tags(line)
                      
        self.doc_text.config(state=tk.DISABLED)
    
    def _adjust_color(self, hex_color, factor):
        # Placeholder for color adjustment if we want dynamic theming later
        return hex_color
