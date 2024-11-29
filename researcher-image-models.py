from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict, BeforeValidator
from typing import List, Optional, Dict, Annotated
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
import instructor
from openai import OpenAI

# Initialize Instructor with OpenAI client
client = instructor.from_openai(OpenAI()

""""""
# Example usage showing field validation and metadata updates:

# Create a new image entry
image = ResearchImage(
    filename="artifact_detail_002.jpg",
    filepath="/data/peru/machu_picchu/2024/",
    technical=TechnicalMetadata(
        format=ImageFormat.TIFF,
        size_bytes=4096000,
        dimensions=(4000, 3000),
        dpi=300,
        color_space="RGB"
    ),
    location=LocationMetadata(
        site_name="Machu Picchu",
        coordinates=(-13.1631, -72.5450),
        country="Peru",
        region="Cusco Region"
    ),
    research=ResearchContext(
        project_name="Incan Architecture Survey",
        study_area="Archaeological Documentation",
        tags=["inca", "stonework", "architecture", "ritual-space"],
        classification="architectural-detail"
    ),
    researcher=Researcher(
        name="Dr. Carlos Rodriguez",
        institution="Universidad Nacional de Arqueología",
        email="c.rodriguez@una.edu.pe",
        orcid="https://orcid.org/0000-0002-1234-5678"
    )
)

# Update custom metadata
image.update_metadata(
    weather_conditions="Sunny",
    equipment_used="Nikon D850",
    conservation_status="Good"
)

# Export to JSONL
jsonl_output = image.to_jsonl()

# Create from natural language description
async def process_new_image():
    description = "High-resolution photo of intricate Incan stonework at Machu Picchu, taken at sunrise"
    new_image = await ResearchImage.from_description(description)
    return new_image
"""""")

class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    RAW = "raw"
    HEIC = "heic"

class LocationMetadata(BaseModel):
    site_name: Annotated[
        str, 
        BeforeValidator(lambda x: x.strip())
    ]
    coordinates: Optional[tuple[float, float]] = Field(
        None,
        description="GPS coordinates as (latitude, longitude)"
    )
    country: str
    region: Optional[str] = None

    @field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        if v is not None:
            lat, lon = v
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError("Invalid GPS coordinates")
        return v

class TechnicalMetadata(BaseModel):
    format: ImageFormat
    size_bytes: Annotated[
        int, 
        Field(gt=0, description="File size in bytes")
    ]
    dimensions: Annotated[
        tuple[int, int],
        Field(description="Image dimensions as (width, height)")
    ]
    dpi: Optional[Annotated[
        int,
        Field(gt=0, description="Dots per inch resolution")
    ]] = None
    color_space: Optional[str] = Field(
        None,
        description="Color space (e.g., RGB, CMYK, Grayscale)"
    )

    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v: tuple[int, int]) -> tuple[int, int]:
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        return v

class ResearchContext(BaseModel):
    project_name: str
    study_area: str
    tags: List[str] = Field(
        default_factory=list,
        description="Keywords describing the image content and context"
    )
    classification: Optional[str] = None
    notes: Optional[str] = None

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        return [tag.strip().lower() for tag in v]

class Researcher(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    institution: str
    email: Annotated[
        str,
        Field(pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    ]
    orcid: Optional[Annotated[
        str,
        Field(pattern=r'^https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{4}$')
    ]] = None

class ResearchImage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: Annotated[
        str,
        Field(min_length=1, pattern=r'^[\w\-. ]+$')
    ]
    filepath: Annotated[
        str,
        Field(min_length=1)
    ]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    technical: TechnicalMetadata
    location: LocationMetadata
    research: ResearchContext
    
    # Researcher Information
    researcher: Researcher
    
    # Additional Fields
    storage_url: Optional[HttpUrl] = None
    backup_urls: List[HttpUrl] = Field(default_factory=list)
    custom_metadata: Dict = Field(default_factory=dict)
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    )
    
    def to_jsonl(self) -> str:
        """Convert the image metadata to JSONL format"""
        return self.model_dump_json(exclude_none=True)
    
    def update_metadata(self, **kwargs):
        """Update the custom metadata dictionary"""
        self.custom_metadata.update(kwargs)
        self.updated_at = datetime.utcnow()
    
    @classmethod
    async def from_description(cls, description: str) -> "ResearchImage":
        """Create a ResearchImage instance from a natural language description"""
        return await client.chat.completions.create(
            model="gpt-4",
            response_model=cls,
            messages=[
                {"role": "user", "content": f"Create a research image entry from this description: {description}"}
            ]
        )

# Example usage
example_image = ResearchImage(
    filename="temple_wall_001.jpg",
    filepath="/data/egypt/edfu/2024/",
    technical=TechnicalMetadata(
        format=ImageFormat.JPEG,
        size_bytes=2048576,
        dimensions=(3000, 2000)
    ),
    location=LocationMetadata(
        site_name="Temple of Edfu",
        coordinates=(24.9776, 32.8729),
        country="Egypt",
        region="Aswan Governorate"
    ),
    research=ResearchContext(
        project_name="Egyptian Temple Documentation",
        study_area="Archaeological Photography",
        tags=["temple", "hieroglyphics", "wall-relief", "ptolemaic-period"],
        classification="architectural"
    ),
    researcher=Researcher(
        name="Dr. Jane Smith",
        institution="University of Archaeology",
        email="j.smith@arch.edu"
    )
)