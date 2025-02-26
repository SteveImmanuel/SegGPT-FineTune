class ShipDetection:
  def __init__(self, probability:float, x_coordinate:int, y_coordinate: int, width:int, height: int):
    self.probability = probability
    self.x_coordinate = x_coordinate
    self.y_coordinate = y_coordinate
    self.width = width
    self.height = height