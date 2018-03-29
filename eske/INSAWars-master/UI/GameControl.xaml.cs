using System;
using System.Globalization;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using INSAWars.Game;
using INSAWars.Units;
using UI.Drawing;

namespace UI
{
    
    /// <summary>
    /// Handles the game's graphics.
    /// </summary>
    public partial class GameControl : UserControl
    {

        #region variables
        /// <summary>
        /// Width of a case's texture, in pixels.
        /// </summary>
        private const int CaseWidth = 96;
        /// <summary>
        /// Height of a case's texture, in pixels.
        /// </summary>
        private const int CaseHeight = 96;

        /// <summary>
        /// The width (in pixels) of the visible part of the map.
        /// </summary>
        private const int VisibleWidth = 960;
        /// <summary>
        /// The height (in pixels) of the visible part of the map.
        /// </summary>
        private const int VisibleHeight = 576;

        /// <summary>
        /// The amount (in pixels) of one shifting operation.
        /// Since we don't display the entire map at any given moment,
        /// we need to shift the visible part of the map.
        /// We decided to move one case at a time.
        /// </summary>
        private const int MoveOffset = CaseWidth;
        /// <summary>
        /// The current X offset.
        /// </summary>
        private int _offsetX = 0;
        /// <summary>
        /// The current Y offset.
        /// </summary>
        private int _offsetY = 0;

        /// <summary>
        /// The number of cases we can display at any given moment on the X axis.
        /// </summary>
        private int CaseCountX = (int)Math.Ceiling((double)VisibleWidth / CaseWidth);
        /// <summary>
        /// The number of cases we can display at any given moment on the Y axis.
        /// </summary>
        private int CaseCountY = (int)Math.Ceiling((double)VisibleHeight / CaseHeight);

        private const string FoodTexture = "FoodSmall";
        private const string IronTexture = "IronSmall";

        private const string StudentSmallTexture = "StudentSmall";
        private const string TeacherSmallTexture = "TeacherSmall";
        private const string HeadSmallTexture = "HeadSmall";
        private const string TeacherBigTexture = "TeacherBig";
        private const string StudentBigTexture = "StudentBig";
        private const string HeadBigTexture = "HeadBig";

        private const string FieldTexture = "FieldBig";
        private const string CityTexture = "CityBig";

        private Game _game;
        private Case _selectedCase;

        private Case _displayInvalidCommandOn;
        private bool _visibleMapMoved;

    #endregion

        #region events
        public delegate void CaseClickedHandler(object sender, CaseClickedEventArgs e);
        public event CaseClickedHandler CaseClicked;

        public delegate void CaseSelectedHandler(object sender, CaseSelectedEventArgs e);
        public event CaseSelectedHandler CaseSelected;
        #endregion

        #region constructor
        public GameControl()
        {
            InitializeComponent();
        }
        #endregion constructor

        #region properties
        public Game Game
        {
            set {
                _game = value;
                Map = _game.Map;
                Height = Map.Size * CaseHeight;
                Width = Map.Size * CaseWidth;
            }
        }

        private Map Map
        {
            get;
            set;
        }

        private int OffsetX
        {
            get { return _offsetX; }
            set
            {
                _offsetX = value;
                _visibleMapMoved = true;
            }
        }

        private int OffsetY
        {
            get { return _offsetY; }
            set
            {
                _offsetY = value;
                _visibleMapMoved = true;
            }
        }
        
        public bool HasSelectedCase
        {
            get
            {
                return _selectedCase != null;
            }
        }

        public Case SelectedCase
        {
            get
            {
                return _selectedCase;
            }

            private set
            {
                var handler = CaseSelected;

                if (handler != null && _selectedCase != value)
                {
                    handler(this, new CaseSelectedEventArgs(value));
                }

                _selectedCase = value;
                InvalidateVisual();
            }
        }

        #endregion properties

        #region methods
        public void SelectCase(Case c)
        {
            SelectedCase = c;
        }

        public void ClearCaseSelection()
        {
            SelectedCase = null;
        }
        #endregion

        #region drawings

        /// <summary>
        /// Displays an animated red rectangle on the given case.
        /// </summary>
        /// <param name="c"></param>
        public void DisplayInvalidCommandOn(Case c)
        {
            _displayInvalidCommandOn = c;
            InvalidateVisual();
        }

        protected override void OnRender(DrawingContext context)
        {
            base.OnRender(context);

            if (Map == null)
            {
                return;
            }

            DrawVisibleMap(context);
            DrawCases(context);

            if (_visibleMapMoved)
            {
                DrawMapPosition(context);
                _visibleMapMoved = false;
            }            

            if (_displayInvalidCommandOn != null)
            {
                DrawInvalidCommand(context);
                _displayInvalidCommandOn = null;
            }
        }

        private void DrawInvalidCommand(DrawingContext context)
        {
            var origin = CaseVisibleOffset(_displayInvalidCommandOn.X, _displayInvalidCommandOn.Y);
            var myDoubleAnimation = new DoubleAnimation();
            myDoubleAnimation.From = 1.0;
            myDoubleAnimation.To = 0.0;
            myDoubleAnimation.Duration = new Duration(TimeSpan.FromSeconds(3));
            
            context.PushOpacity(1.0, myDoubleAnimation.CreateClock());
            context.DrawRectangle(new SolidColorBrush((Color)ColorConverter.ConvertFromString("#55FF2B2B")),
                null, new Rect(origin.Item1, origin.Item2, CaseWidth, CaseHeight));
            context.Pop();
        }

        private void DrawMapPosition(DrawingContext context)
        {
            var myDoubleAnimation = new DoubleAnimation();
            myDoubleAnimation.From = 1.0;
            myDoubleAnimation.To = 0.0;
            myDoubleAnimation.Duration = new Duration(TimeSpan.FromSeconds(3));
         
            context.PushOpacity(1.0, myDoubleAnimation.CreateClock());
            context.DrawRectangle(new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF2B2B2B")),
                                  null, new Rect(702, 10, 248, 64));
            var relativePositions = RelativePositions();
            var position = new FormattedText(
                        relativePositions.Item1.ToString() + "% / " + relativePositions.Item2.ToString() + "%",
                        CultureInfo.GetCultureInfo("en-us"),
                        FlowDirection.LeftToRight,
                        new Typeface("Charlemagne STD"),
                        36,
                        new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF9C9C8F")));
            context.DrawText(position, new Point(712, 20));
            context.Pop();
        }

        /// <summary>
        /// Draws the visible portion of the map.
        /// </summary>
        /// <returns></returns>
        private void DrawVisibleMap(DrawingContext context)
        {
            var rect = new Rect(0, 0, CaseWidth, CaseHeight);
            var origin = TopLeftCaseIndexes(OffsetX, OffsetY);

            for (int i = 0; i < CaseCountX; i++)
            {
                for (int j = 0; j < CaseCountY; j++)
                {
                    rect.X = i * CaseWidth;
                    rect.Y = j * CaseHeight;

                    context.DrawImage((BitmapImage)FindResource(Map.GetCaseAt(i + origin.Item1, j + origin.Item2).Texture), rect);
                }
            }
        }

        private void DrawCases(DrawingContext context)
        {
            var origin = TopLeftCaseIndexes(OffsetX, OffsetY);

            for (int i = 0; i < CaseCountX; i++)
            {
                for (int j = 0; j < CaseCountY; j++)
                {
                    var c = Map.GetCaseAt(i + origin.Item1, j + origin.Item2);
                    DrawCaseContent(context, c, CaseOffset(i, j));

                    if (c == SelectedCase)
                    {
                        DrawCaseSelection(context, c, CaseOffset(i, j));
                    }
                }
            }
        }

        private void DrawCaseSelection(DrawingContext context, Case c, Tuple<int, int> origin)
        {
            context.DrawRectangle(null,
                new Pen(new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF2B2B2B")), 2), new Rect(origin.Item1, origin.Item2, CaseWidth, CaseHeight));
        }

        private void DrawCaseContent(DrawingContext context, Case c, Tuple<int, int> origin)
        {
            var drawer = new CaseDrawer(context, origin);
            DrawFood(c.Food, drawer);
            DrawIron(c.Iron, drawer);

            if (_game.IsVisible(c))
            {
                if (c.HasCity)
                {
                    DrawCity(drawer);
                }
                else if (c.IsUsed)
                {
                    DrawField(drawer);
                }
                else
                {
                    DrawMostDefensiveUnit(c, drawer);                    
                }
                DrawUnits(c, drawer);
            }
            else { 
                DrawFogOfWar(context, origin);
            } 
        }

        private void DrawMostDefensiveUnit(Case c, CaseDrawer drawer)
        {
            if (c.HasUnits)
            {
                drawer.DrawMainItem((BitmapImage)FindResource(c.MostDefensiveUnit.Texture + "Big"));
            }
        }

        private void DrawField(CaseDrawer drawer)
        {
            drawer.DrawMainItem((BitmapImage)FindResource(FieldTexture));
        }

        private void DrawCity(CaseDrawer drawer)
        {
            drawer.DrawMainItem((BitmapImage)FindResource(CityTexture));
        }

        private void DrawUnits(Case c, CaseDrawer drawer)
        {
            DrawStudents(c.Students, drawer);
            DrawTeachers(c.Teachers, drawer);
        }

        private void DrawStudents(IEnumerable<Unit> students, CaseDrawer drawer)
        {
            if (students.Count() > 0)
            {
                var formattedText = new FormattedText(
                        students.Count().ToString(),
                        CultureInfo.GetCultureInfo("en-us"),
                        FlowDirection.LeftToRight,
                        new Typeface("Charlemagne STD"),
                        12,
                        Brushes.White);
                drawer.Draw((BitmapImage)FindResource(StudentSmallTexture), formattedText);
            }
        }

        private void DrawTeachers(IEnumerable<Unit> teachers, CaseDrawer drawer)
        {
            if (teachers.Count() > 0)
            {
                var formattedText = new FormattedText(
                        teachers.Count().ToString(),
                        CultureInfo.GetCultureInfo("en-us"),
                        FlowDirection.LeftToRight,
                        new Typeface("Charlemagne STD"),
                        12,
                        Brushes.White);
                drawer.Draw((BitmapImage)FindResource(TeacherSmallTexture), formattedText);
            }
        }

        private void DrawFood(int amount, CaseDrawer drawer)
        {
            if (amount > 0)
            {         
                drawer.Draw((BitmapImage)FindResource(FoodTexture));
            }
        }
        
        private void DrawIron(int amount, CaseDrawer drawer)
        {
            if (amount > 0)
            {
                drawer.Draw((BitmapImage)FindResource(IronTexture));
            }
        }

        private void DrawFogOfWar(DrawingContext context, Tuple<int, int> origin)
        {
            context.DrawRectangle(new SolidColorBrush((Color)ColorConverter.ConvertFromString("#552B2B2B")),
                                  null, new Rect(origin.Item1, origin.Item2, CaseWidth, CaseHeight));
        }

        /// <summary>
        /// Returns the pixel offsets for the given case's indexes.
        /// </summary>
        /// <param name="i"></param>
        /// <param name="j"></param>
        /// <returns></returns>
        private Tuple<int, int> CaseOffset(int i, int j)
        {
            return Tuple.Create(i * CaseWidth, j * CaseHeight);
        }

        private Tuple<int, int> CaseVisibleOffset(int i, int j)
        {
            var origin = CaseOffset(i, j);
            return Tuple.Create(origin.Item1 - OffsetX, origin.Item2 - OffsetY);
        }

        /// <summary>
        /// Determines the indexes of the top left case according to the given offset.
        /// </summary>
        /// <param name="offsetX">The current pixel X offset</param>
        /// <param name="offsetY">The current pixel Y offset</param>
        /// <returns>A tuple of indices for the map array corresponding to the origin case.</returns>
        private Tuple<int, int> TopLeftCaseIndexes(int offsetX, int offsetY)
        {
            return Tuple.Create(offsetX / CaseWidth,
                                       offsetY / CaseHeight);
        }

        /// <summary>
        /// Determines the current relative positions of the map.
        /// </summary>
        /// <returns></returns>
        private Tuple<int, int> RelativePositions()
        {
            return Tuple.Create((int)((double)OffsetX * 100 / (Width - (CaseCountX * CaseWidth))),
                                       (int)((double)OffsetY * 100 / (Height - (CaseCountY * CaseHeight))));
        }
        
        #endregion

        #region events
        /// <summary>
        /// Handles mouse left clicks.
        /// <param name="e"></param>
        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left)
            {
                double x = e.GetPosition(this).X;
                double y = e.GetPosition(this).Y;
                Case c = CaseAtPosition(x + OffsetX, y + OffsetY);

                var handler = CaseClicked;

                if (handler != null && _selectedCase != c)
                {
                    handler(this, new CaseClickedEventArgs(c));
                }

                FocusManager.SetFocusedElement(Application.Current.MainWindow, this);
                InvalidateVisual();
            }            
        }

        /// <summary>
        /// Moves the visible map to make sure the selected case is visible.
        /// </summary>
        public void MoveVisibleMapToSelectedCase()
        {
            if (_selectedCase != null)
            {
                OffsetX = Math.Max(0, _selectedCase.X - (CaseCountX - 1) / 2) * CaseWidth;
                OffsetX = Math.Min(CaseWidth * (Map.Size - CaseCountX), OffsetX);
                OffsetY = Math.Max(0, _selectedCase.Y - (CaseCountY - 1) / 2) * CaseHeight;
                OffsetY = Math.Min(CaseHeight * (Map.Size - CaseCountY), OffsetY);

                InvalidateVisual();
            }
        }

        /// <summary>
        /// Handles mouse wheel to move the visible map vertically.
        /// </summary>
        /// <param name="e"></param>
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            if (e.Delta > 0)
            {
                if (CanMoveVisibleMapUp())
                {
                    MoveVisibleMapUp();
                }
            }
            else
            {
                if (CanMoveVisibleMapDown())
                {
                    MoveVisibleMapDown();
                }
            }
        }
        
        #endregion

        #region visible map handling
        public bool CanMoveVisibleMapLeft()
        {
            return OffsetX > 0;
        }

        public bool CanMoveVisibleMapRight()
        {
            return OffsetX + (CaseCountX * CaseWidth) <= Width - CaseWidth;
        }

        public bool CanMoveVisibleMapUp()
        {
            return OffsetY > 0;
        }

        public bool CanMoveVisibleMapDown()
        {
            return OffsetY + (CaseCountY * CaseHeight) <= Height - CaseHeight;
        }

        public void MoveVisibleMapLeft()
        {
            OffsetX -= MoveOffset;
            InvalidateVisual();
        }

        public void MoveVisibleMapRight()
        {
            OffsetX += MoveOffset;
            InvalidateVisual();
        }

        public void MoveVisibleMapUp()
        {
            OffsetY -= MoveOffset;
            InvalidateVisual();
        }

        public void MoveVisibleMapDown()
        {
            OffsetY += MoveOffset;
            InvalidateVisual();
        }

        /// <summary>
        /// Returns the case at the given position on the screen (relative to the game's window).
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        private Case CaseAtPosition(double x, double y)
        {
            int i = (int)(x / CaseWidth);
            int j = (int)(y / CaseHeight);
            return Map.GetCaseAt(i, j);
        }

        #endregion

       
    }
}
