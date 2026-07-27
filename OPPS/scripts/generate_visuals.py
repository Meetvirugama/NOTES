import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Create Visuals directory
os.makedirs("../notes/Visuals", exist_ok=True)

def setup_plot(title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    return fig, ax, filename

def save_plot(fig, filename):
    plt.tight_layout()
    plt.savefig(f"../notes/Visuals/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', text_color='black', title=None):
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='#1565C0', facecolor=color, zorder=2)
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 0.5, title, ha='center', va='center', fontsize=12, fontweight='bold', color=text_color, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.5, text, ha='center', va='center', fontsize=11, color=text_color, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=12, fontweight='bold', color=text_color, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate(text, xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8),
                fontsize=10, ha='center', va='center', zorder=1)

# 1. OOP Overview
fig, ax, fname = setup_plot("The 4 Pillars of OOP", "01_oop_overview.png")
draw_box(ax, 3.5, 7, 3, 2, "OOP System")
draw_box(ax, 1, 3, 3, 2, "Encapsulation\n(Data Hiding)", color="#E8F5E9", text_color="#2E7D32")
draw_box(ax, 6, 3, 3, 2, "Abstraction\n(Hide Complexity)", color="#FFF3E0", text_color="#E65100")
draw_box(ax, 1, 0, 3, 2, "Inheritance\n(Code Reuse)", color="#F3E5F5", text_color="#6A1B9A")
draw_box(ax, 6, 0, 3, 2, "Polymorphism\n(Many Forms)", color="#E0F7FA", text_color="#006064")
draw_arrow(ax, 5, 7, 2.5, 5)
draw_arrow(ax, 5, 7, 7.5, 5)
draw_arrow(ax, 5, 7, 2.5, 2)
draw_arrow(ax, 5, 7, 7.5, 2)
save_plot(fig, fname)

# 2. Classes and Objects
fig, ax, fname = setup_plot("Classes vs Objects (Memory Layout)", "02_classes_objects.png")
draw_box(ax, 1, 4, 3, 4, "Logical Blueprint\n(Code Segment)", title="CLASS (Car)")
draw_box(ax, 6, 6, 3, 3, "Physical Entity\n(Heap Memory)", title="OBJECT 1 (Red Car)")
draw_box(ax, 6, 2, 3, 3, "Physical Entity\n(Heap Memory)", title="OBJECT 2 (Blue Car)")
draw_arrow(ax, 4, 6, 6, 7.5, "Instantiates")
draw_arrow(ax, 4, 6, 6, 3.5, "Instantiates")
save_plot(fig, fname)

# 3. Inheritance
fig, ax, fname = setup_plot("Inheritance (IS-A Relationship)", "03_inheritance.png")
draw_box(ax, 3.5, 6, 3, 3, "+ speed\n+ color\n+ start()", title="Vehicle (Base)")
draw_box(ax, 1, 1, 3, 3, "+ trunkSize\n+ drift()", title="Car (Derived)")
draw_box(ax, 6, 1, 3, 3, "+ hasKickstand\n+ wheelie()", title="Bike (Derived)")
draw_arrow(ax, 2.5, 4, 5, 6, "Inherits")
draw_arrow(ax, 7.5, 4, 5, 6, "Inherits")
save_plot(fig, fname)

# 4. Polymorphism
fig, ax, fname = setup_plot("Polymorphism (Method Overriding)", "04_polymorphism.png")
draw_box(ax, 1, 4, 8, 2, "Animal a = new Dog();\n\na.makeSound();", color="#FFF9C4")
draw_box(ax, 1, 1, 3, 2, "Output: 'Woof!'", color="#E8F5E9")
draw_box(ax, 6, 1, 3, 2, "Resolved at Runtime\n(Dynamic Binding)")
draw_arrow(ax, 5, 4, 2.5, 3)
draw_arrow(ax, 5, 4, 7.5, 3)
save_plot(fig, fname)

# 5. Encapsulation
fig, ax, fname = setup_plot("Encapsulation (Data Hiding)", "05_encapsulation.png")
draw_box(ax, 2.5, 3, 5, 5, "", title="Class (Capsule)")
draw_box(ax, 3.5, 4, 3, 1, "private int data", color="#FFCDD2")
draw_box(ax, 3.5, 6, 3, 1, "public get/set()", color="#C8E6C9")
draw_arrow(ax, 5, 6, 5, 5, "Controls Access")
draw_arrow(ax, 0.5, 6.5, 3.5, 6.5, "Client")
save_plot(fig, fname)

# 6. V-Table (Polymorphism Internals)
fig, ax, fname = setup_plot("C++ V-Table Layout (Under the Hood)", "06_vtable_internals.png")
draw_box(ax, 1, 6, 3, 2, "v-ptr (Hidden)\n\n+ age = 5", title="Dog Object", color="#E8F5E9")
draw_box(ax, 6, 5, 3, 4, "[0] ~Dog()\n[1] Dog::sound()\n[2] Animal::sleep()", title="Dog v-table", color="#FFF3E0")
draw_arrow(ax, 2.5, 7.5, 5.8, 8, "v-ptr points to class table")
save_plot(fig, fname)

# 7. CPU Cache (OOP vs DOD)
fig, ax, fname = setup_plot("OOP vs Data-Oriented Design (CPU Cache)", "07_cpu_cache.png")
draw_box(ax, 1, 6, 8, 2, "P1(x,y) | P1(vel) | P2(x,y) | P2(vel)", title="OOP (Array of Structs)", color="#FFCDD2")
draw_box(ax, 1, 2, 8, 2, "P1(vel) | P2(vel) | P3(vel) | P4(vel)", title="DOD (Struct of Arrays)", color="#C8E6C9")
ax.text(5, 5.5, "Updating Velocities only -> High Cache Misses!", ha='center', color='red', fontweight='bold')
ax.text(5, 1.5, "Updating Velocities only -> 100% Cache Hit!", ha='center', color='green', fontweight='bold')
save_plot(fig, fname)

print("✅ OOP visuals successfully generated in the Visuals/ folder using matplotlib!")
