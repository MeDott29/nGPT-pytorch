from manim import *
import numpy as np

class L2NormalizationScene(ThreeDScene):
    def construct(self):
        # Configure the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Create the axes
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            z_length=4
        )
        
        # Create the unit sphere
        sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(24, 48),
            fill_opacity=0.2,
            stroke_opacity=0.2,
            fill_color=BLUE,
            stroke_color=BLUE
        )
        
        # Initial setup
        self.play(Create(axes), Create(sphere))
        self.wait()
        
        # Function to create vector and its components
        def create_vector(coords, color=RED):
            vector = Arrow3D(
                start=np.array([0, 0, 0]),
                end=np.array(coords),
                color=color
            )
            components = VGroup(*[
                Line3D(
                    start=np.array([0, 0, 0]),
                    end=np.array([
                        coords[0] if i == 0 else 0,
                        coords[1] if i == 1 else 0,
                        coords[2] if i == 2 else 0
                    ]),
                    color=GRAY,
                    stroke_opacity=0.5
                )
                for i in range(3)
            ])
            return vector, components
        
        # Original vector
        original_coords = np.array([1.5, 1.0, 0.8])
        original_vector, original_components = create_vector(original_coords)
        
        # Add original vector
        self.play(Create(original_components), Create(original_vector))
        self.wait()
        
        # Calculate normalized coordinates
        magnitude = np.linalg.norm(original_coords)
        normalized_coords = original_coords / magnitude
        
        # Create normalized vector
        normalized_vector, normalized_components = create_vector(normalized_coords, color=GREEN)
        
        # Show normalization process
        # First show magnitude calculation
        magnitude_line = Line3D(
            start=np.array([0, 0, 0]),
            end=original_coords,
            color=YELLOW,
            stroke_width=5
        )
        magnitude_text = Text(
            f"Magnitude = {magnitude:.2f}",
            font_size=24
        ).to_corner(UL)
        
        self.play(
            Create(magnitude_line),
            Write(magnitude_text)
        )
        self.wait()
        
        # Show normalization
        normalize_text = Text(
            "Normalizing: v → v/|v|",
            font_size=24
        ).next_to(magnitude_text, DOWN)
        
        self.play(Write(normalize_text))
        self.play(
            Transform(original_vector, normalized_vector),
            Transform(original_components, normalized_components),
            Transform(magnitude_line, Line3D(
                start=np.array([0, 0, 0]),
                end=normalized_coords,
                color=YELLOW,
                stroke_width=5
            ))
        )
        self.wait()
        
        # Show norm_eps shell
        eps = 0.1
        outer_sphere = Surface(
            lambda u, v: (1 + eps) * np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(24, 48),
            fill_opacity=0.1,
            stroke_opacity=0.1,
            fill_color=BLUE_E,
            stroke_color=BLUE_E
        )
        
        inner_sphere = Surface(
            lambda u, v: (1 - eps) * np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(24, 48),
            fill_opacity=0.1,
            stroke_opacity=0.1,
            fill_color=BLUE_E,
            stroke_color=BLUE_E
        )
        
        eps_text = Text(
            f"norm_eps = {eps}",
            font_size=24
        ).next_to(normalize_text, DOWN)
        
        self.play(
            Write(eps_text),
            Create(outer_sphere),
            Create(inner_sphere)
        )
        self.wait(2)
        
        # Clean up
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )
        self.wait()

if __name__ == "__main__":
    from manim.utils.file_ops import guarantee_existence
    guarantee_existence("media")
    scene = L2NormalizationScene()
    scene.render()
