from tkinter import filedialog, Button, Tk

def open_file_dialog():
    
    file_path = None

    def open_image():
        nonlocal file_path
        path = filedialog.askopenfilename(
            initialdir="C:\\Users\\PC\\Desktop\\Vezbanje python\\Images\\diplomski_pocetno\\dataset",
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg")]
        )
        if path:
            file_path = path
        root.destroy()

    root = Tk()
    root.title("Select Image")
    Button(root, text="Choose Image", command=open_image).pack()
    Button(root, text="Cancel", command=root.destroy).pack()
    root.mainloop()

    return file_path

#https://pastebin.com/Pcg75LxB tutorijal
#https://www.youtube.com/watch?v=q8WDvrjPt0M
