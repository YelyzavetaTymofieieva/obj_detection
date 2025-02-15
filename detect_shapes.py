from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional
import math
import pytest 

# Sample input:
# Square Side 1 TopRight 1 1 
# Rectangle TopRight 2 2 BottomLeft 1 1
# Circle Center 1 1 Radius 2

# Sample output:
# Square Perimeter 4 Area 1
# Rectangle Perimeter 4 Area 1
# Circle Perimeter 1 Area 2

class Shape(Enum):
    CIRCLE = 1
    SQUARE = 2
    RECTANGLE = 3
    TRIANGLE = 4
    
    def __str__(self) -> str:
        return self.name.capitalize()
    
    def parse_shape(input_str: str) -> "Shape": # Iterate over all possible Shape values
        for shape in list(Shape): #Check if input_str matches a Shape string
            if input_str == str(shape):
                return shape # Return the matching Shape
        raise ValueError(f"invalid input for shape: {input_str}")



@dataclass 
class Datapoint:
    @classmethod
    def try_parse(cls, input_str: str) -> Optional[tuple["Datapoint", str]]: # parse a Side object from a string.
        """returns a tuple containing an obj of type Datapoint and the remaining string
        or None, if we could not parse that object from the string."""
        name = cls.get_name()
        num_args = cls.get_num_args()

        if input_str.startswith(name):
            # e.g. parts = ["Side", "3.5", "ExtraText"]
            parts = input_str.split(" ", maxsplit=1 + num_args) #Split the Input into Parts 
            assert len(parts) >= (1 + num_args), f"Parsing {name} failed. Got parts {parts}"

            name_part, tail_parts = parts[0], parts[1:] # name_part should match name (e.g. "Side") 
            assert name_part == name, f"Invalid input to datapoint type {name}: {name_part}"

            args = [float(arg_str) for arg_str in tail_parts[:num_args]]
            #turn this part of the split string into a float for future manipulations
            if len(tail_parts) == num_args:
                # There is no remainder from splitting the string into 1 (name) + num_args
                # Hence, we return an empty string for the remainder
                return cls(*args), ""
                 #cls(*args) dynamically creates an instance of the class
            else:
                assert len(tail_parts) == num_args + 1
                return cls(*args), tail_parts[-1]
        return None
    
    @classmethod
    def get_name(cls) -> str: #returns the name of the class as a string
        return cls.__name__
    """
    This is useful when parsing because it ensures that the input string starts with the expected name of the Datapoint subclass (e.g., Side).
    # Example: If cls refers to Side, then name = "Side".
    """
    
    @classmethod
    def get_num_args(cls) -> int:
        return len(fields(cls))
    
    @staticmethod
    def parse_datapoints(shape: Shape, input_str: str) -> list["Datapoint"]:
        datapoint_types = SHAPE_TO_VALID_DATAPOINT_TYPES[shape]
        datapoints = []  #initialize an empty list that will store the successfully parsed Datapoint objects
        while len(input_str) > 0: # the while loop will continue parsing until the entire input_str is consumed
            for datapoint_type in datapoint_types: #iterate over each datapoint_type in the datapoint_types list (e.g., [Side])
                result = datapoint_type.try_parse(input_str) 
                if result is None:
                    continue
                datapoint, input_str = result
                assert isinstance(datapoint, datapoint_type)
                datapoints.append(datapoint)
                break
            else:
                if len(input_str) > 0:
                    raise ValueError(
                        f"Could not parse input datapoints for shape {shape}: {input_str}"
                    )
        return datapoints
    
    
    # e.g. Datapoint.parse_datapoints(Shape.TRIANGLE, "Side 5 Angle 90")
    # e.g. output [Side(5.0), Angle(90.0)]

# These subclasses allow us to parse and structure geometric information from strings.
 #The subclasses below represent different types of data points that can be 
    # extracted from an input string and later used in geometric or spatial 
    # computations. Let's break down their purpose and relationships.

@dataclass
class Angle(Datapoint):
    value: float
                
@dataclass
class Side(Datapoint):
    value: float

@dataclass
class RectangularCoordinate(Datapoint):
    x: float
    y: float

@dataclass
class TopRight(RectangularCoordinate):
    pass

@dataclass
class TopLeft(RectangularCoordinate):
    pass

@dataclass
class BottomRight(RectangularCoordinate):
    pass

@dataclass
class BottomLeft(RectangularCoordinate):
    pass

@dataclass
class Radius(Datapoint):
    value: float

RECTANGULAR_COORDINATES = [TopRight, TopLeft, BottomLeft, BottomRight]

SHAPE_TO_VALID_DATAPOINT_TYPES: dict[Shape, list[Datapoint]] = {
    Shape.CIRCLE: [Radius],
    Shape.SQUARE: [Side] + RECTANGULAR_COORDINATES,
    Shape.RECTANGLE: [Side] + RECTANGULAR_COORDINATES,
    Shape.TRIANGLE: [Side, Angle]
}


@dataclass
class Solution:
    shape: Shape
    perimeter: float
    area: float
    
    def __str__(self) -> str:
        """converts a Solution object to a readable string format"""
        return f"{str(self.shape)} Perimeter {self.perimeter} Area {self.area}"

@dataclass
class Problem:
    shape: Shape
    datapoints: list[Datapoint]
    
    @staticmethod
    def parse_problem(input_str: str) -> "Problem":
        try:
            shape_str, datapoints = input_str.split(" ", 1)
        except Exception:
            raise ValueError("no datapoints provided")
        shape = Shape.parse_shape(shape_str)
        return Problem(
            shape=shape,
            datapoints=Datapoint.parse_datapoints(shape, datapoints)
        )
    
    def solve(self) -> Solution:  
        if self.shape == Shape.SQUARE:
            return solve_square(self.datapoints)
        if self.shape == Shape.RECTANGLE:
            return solve_rect_from_opposing_coord(self.datapoints)
        if self.shape == Shape.CIRCLE:
            return solve_circle_from_r(self.datapoints)
        if self.shape == Shape.TRIANGLE:
            return solve_triangle(self.datapoints)
        raise ValueError(f"Could not solve for shape: {self.shape}")


def solve_square(datapoints: list[Datapoint]) -> Solution:
    if (solution := solve_square_from_side(datapoints)) is not None:
        return solution
    if (solution := solve_square_from_rectangular_coordinates(datapoints)) is not None:
        return solution
            
    raise ValueError(f"Could not solve for Square with datapoints {datapoints}")


def solve_triangle(datapoints: list[Datapoint])-> Optional[Solution]:
    angle: Optional[Angle] = None
    sides: list[Side] = []
    for datapoint in datapoints:
        if isinstance(datapoint, Side):
            sides.append(datapoint)
        if isinstance(datapoint, Angle):
            angle = datapoint
            
    if len(sides) < 2 or angle is None:
        return None

    side1 = sides[0].value
    side2 = sides[1].value
    angle_degrees = angle.value
    angle_radians = math.radians(angle_degrees)
    # Calculate the third side using the law of cosines
        # c = √(a² + b² - 2ab × cos(C))
    side3 = math.sqrt(side1**2 + side2**2 - 2 * side1 * side2 * math.cos(angle_radians))

    # Calculate perimeter
    perimeter = round(side1 + side2 + side3,2)

    # Calculate area using the formula: 0.5 * a * b * sin(angle)
    area = round(0.5 * side1 * side2 * math.sin(angle_radians), 2)

    return Solution(shape = Shape.TRIANGLE, perimeter=perimeter, area=area)


def solve_square_from_side(datapoints: list[Datapoint]) -> Optional[Solution]:
    for datapoint in datapoints:
        if isinstance(datapoint, Side):
            return Solution(
                shape=Shape.SQUARE,
                perimeter=4 * datapoint.value,
                area=datapoint.value * datapoint.value,
            )
    return None

def solve_square_from_rectangular_coordinates(datapoints: list[Datapoint]) -> Optional[Solution]:
    # For square, I can solve both from adjacent and opposite coordinates
    if (solution := solve_square_from_adjacent_coordinates(datapoints)) is not None:
        return solution
    return None

def validate_unique_rectangular_coordinates(first: RectangularCoordinate, second: RectangularCoordinate):
    assert not (first.x == second.x and first.y == second.y)
    if first.y > second.y:
        assert first.get_name().startswith("Top")
        assert second.get_name().startswith("Bottom")
    if second.y > first.y:
        assert second.get_name().startswith("Top")
        assert first.get_name().startswith("Bottom")  
    if first.x > second.x:
        assert first.get_name().endswith("Right")
        assert second.get_name().endswith("Left")
    if second.x > first.x:
        assert second.get_name().endswith("Right")
        assert first.get_name().endswith("Left")    

def solve_square_from_adjacent_coordinates(datapoints: list[Datapoint]) -> Optional[Solution]:
    rec_coords: list[RectangularCoordinate] = [
        d for d in datapoints if isinstance(d, RectangularCoordinate)
    ]
    # Pairwise comparison
    for (i, first) in enumerate(rec_coords):
        for second in rec_coords[i + 1:]:
            validate_unique_rectangular_coordinates(first, second)
            if first.x == second.x:
                side = abs(first.y - second.y) 
            if first.y == second.y:
                side = abs(first.x - second.x)
            assert side > 0, "Side must be greater than 0"
            return solve_square_from_side([Side(value=side)])
    return None

# def solve_from_opposing_coordinates(shape: Shape, datapoints: list[Datapoint]) -> Optional[Solution]:
def solve_rect_from_opposing_coord(datapoints: list[Datapoint]) -> Optional[Solution]:
    rect_coords: list[RectangularCoordinate] = [
        d for d in datapoints if isinstance (d, RectangularCoordinate)
    ]
    for (i, first) in enumerate(rect_coords):
        for second in rect_coords[i+1:]:
            validate_unique_rectangular_coordinates(first, second)
            if first.x == second.x or first.y == second.y:
                # check if the coordinates are adjacent and skip them because they aren't opposite 
                continue
            side_1 = abs(second.x - first.x)
            side_2 = abs(second.y - first.y)
            shape = Shape.RECTANGLE
            if shape == Shape.SQUARE:
                assert side_1 == side_2 # type: ignore
            return Solution(shape=shape, perimeter= 2* side_1 + 2* side_2, area= side_1 * side_2)
    return None
    
# def solve_circle_from_radius(datapoints: list[Datapoint]) -> Optional[Solution]:
def solve_circle_from_r(datapoints: list[Datapoint]) -> Optional[Solution]:
    for datapoint in datapoints:
        if isinstance(datapoint, Radius):
            return Solution(
                shape=Shape.CIRCLE,
                perimeter= round(2*math.pi * datapoint.value, 2),
                area= round(datapoint.value *datapoint.value *math.pi, 2),
            )
    return None

# unit tests
def test_solution_to_string():
    solution_1 = Solution(
        shape=Shape.CIRCLE,
        area=2.0,
        perimeter=10.0,
    )
    solution_2 = Solution(
        shape = Shape.RECTANGLE,
        perimeter = 5.0,
        area = 4.0,
    )
    assert str(solution_1) == "Circle Perimeter 10.0 Area 2.0",\
        str(solution_1)
    
    assert str(solution_2) == "Rectangle Perimeter 5.0 Area 4.0",\
        str(solution_2)
 
 
# unit test for try_parse       
def test_try_parse():
    assert Side.try_parse("Side 1 TopRight 1 1") == (Side(1), "TopRight 1 1")
    assert TopRight.try_parse("Side 1 TopRight 1 1") == None
    assert Radius.try_parse("TopRight 1 1 Radius 4") == None
    

# unit test for parse_datapoints
def test_parse_datapoints():
    assert Datapoint.parse_datapoints(Shape.SQUARE, "Side 4") == \
        [Side(4)]
    assert Datapoint.parse_datapoints(Shape.RECTANGLE, "TopRight 2 1 BottomLeft -2 -1") == \
        [TopRight(2,1), BottomLeft(-2, -1)]
    assert Datapoint.parse_datapoints(Shape.CIRCLE, "Radius 2") == \
        [Radius(2)]
        
def test_fail_to_parse_datapoints():
    with pytest.raises(
        ValueError,
        match="Could not parse input datapoints for shape Circle: Side 2"
    ):
        Datapoint.parse_datapoints(Shape.CIRCLE, "Side 2")
      
# unit test for parse_problem   

def test_parse_problem():
    assert Problem.parse_problem("Square Side 4") == Problem(Shape.SQUARE,[Side(4)]) 
    assert Problem.parse_problem(
        "Rectangle TopRight 2 1 BottomLeft -2 -1"
    ) == Problem(Shape.RECTANGLE,[TopRight(2,1), BottomLeft(-2,-1)])
    assert Problem.parse_problem(
        "Circle Radius 3"
    ) == Problem(Shape.CIRCLE,[Radius(3)])  

def test_parse_shape():
    assert Shape.parse_shape("Circle") == Shape.CIRCLE
    assert Shape.parse_shape("Square") == Shape.SQUARE
    assert Shape.parse_shape("Rectangle") == Shape.RECTANGLE
# unit test for solve  

def test_solve():
    assert Problem(Shape.SQUARE,[Side(5)]).solve() == \
        Solution(Shape.SQUARE, perimeter=20, area=25)
    
    assert Problem(Shape.CIRCLE, [Radius(2)]).solve() == \
        Solution(Shape.CIRCLE, perimeter=12.57, area=12.57)
    
    assert Problem(Shape.RECTANGLE,[TopRight(2,1),BottomLeft(-2, -1)]).solve() == \
        Solution(Shape.RECTANGLE, perimeter= 12, area= 8) 
    
# unit test for solve_sq from rect coord 
def test_solve_square_from_side():
    assert solve_square_from_side([Side(2)]) == \
    Solution(Shape.SQUARE, perimeter= 8, area= 4)
    
    assert solve_square_from_side([Radius(1)]) == None
    
# unit test for solve sq from adj coord

def test_solve_square_from_adjacent_coordinates():
   assert solve_square_from_adjacent_coordinates([TopRight(1,1), BottomRight(1,-1)]) == \
       Solution(Shape.SQUARE, perimeter= 8, area= 4)

# unit test for solve from opposing coord         
def test_solve_rect_from_opposing_coordinates():
    assert solve_rect_from_opposing_coord([TopRight(2,1), BottomLeft(-2,-1)]) == \
    Solution(Shape.RECTANGLE, perimeter=12, area= 8)
    
    assert solve_rect_from_opposing_coord([Side(4), TopLeft(2,1)]) == None

# unit test for solve circle  
def test_solve_circle_from_r():
    assert solve_circle_from_r([Radius(2)]) == \
        Solution(shape= Shape.CIRCLE, perimeter= 12.57, area= 12.57)
    
    assert solve_circle_from_r([Side(2)]) == None
 
def input_to_output(input_str) -> str:
    problem = Problem.parse_problem(input_str)
    solution = problem.solve(problem)
    return str(solution)
    
def run_tests():
    test_solution_to_string()
    test_parse_shape()
    test_parse_problem()
    test_try_parse()
    test_parse_datapoints()
    test_fail_to_parse_datapoints()
    # Solving methods
    test_solve_circle_from_r()
    test_solve_square_from_side()
    test_solve_square_from_adjacent_coordinates()
    test_solve_rect_from_opposing_coordinates()
    
    print("All tests pass!")
 
if __name__ == "__main__":
    # print(str(Shape.CIRCLE), str(Shape.RECTANGLE))
    run_tests()
    print("Enter your problem:")
    input_str = input()
    problem = Problem.parse_problem(input_str)
    solution = problem.solve()
    print(str(solution))
    
            