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
    /// Interaction logic for GameView.xaml
    /// </summary>
    public partial class GameView : UserControl
    {
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
        private int OffsetX = 0;
        /// <summary>
        /// The current Y offset.
        /// </summary>
        private int OffsetY = 0;

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
        private const string StudentTexture = "StudentSmall";
            private const string TeacherTexture = "TeacherSmall";

        private Map _map;

        #region constructor
        public GameView()
        {
            InitializeComponent();
        }
        #endregion constructor

        #region properties
        public Map Map
        {
            get { return _map; }
            set {
                _map = value;
                Height = Map.Size * CaseHeight;
                Width = Map.Size * CaseWidth;
            }
        }
        #endregion properties

        #region drawings

        protected override void OnRender(DrawingContext context)
        {
            base.OnRender(context);
            DrawVisibleMap(context);
            DrawCases(context);
            DrawMapPosition(context);
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
                        Brushes.PaleVioletRed);
            context.DrawText(position, new Point(712, 20));
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
                    DrawCaseContent(context, Map.GetCaseAt(i + origin.Item1, j + origin.Item2), CaseOffset(i, j));
                }
            }
        }

        private void DrawCaseContent(DrawingContext context, Case c, Tuple<int, int> origin)
        {
            var drawer = new CaseDrawer(context, origin);
            DrawFood(c.Food, drawer);
            DrawIron(c.Iron, drawer);
            DrawUnits(c, drawer);
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
                        Brushes.PaleVioletRed);
                drawer.Draw(formattedText, (BitmapImage)FindResource(StudentTexture));
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
                        Brushes.PaleVioletRed);
                drawer.Draw(formattedText, (BitmapImage)FindResource(TeacherTexture));
            }
        }

        private void DrawFood(int amount, CaseDrawer drawer)
        {
            if (amount > 0)
            {         
                var formattedText = new FormattedText(
                    amount.ToString(),
                    CultureInfo.GetCultureInfo("en-us"),
                    FlowDirection.LeftToRight,
                    new Typeface("Charlemagne STD"),
                    12,
                    Brushes.PaleVioletRed);
                drawer.Draw(formattedText, (BitmapImage)FindResource(FoodTexture));
            }
        }
        
        private void DrawIron(int amount, CaseDrawer drawer)
        {
            if (amount > 0)
            {
                var formattedText = new FormattedText(
                    amount.ToString(),
                    CultureInfo.GetCultureInfo("en-us"),
                    FlowDirection.LeftToRight,
                    new Typeface("Charlemagne STD"),
                    12,
                    Brushes.Red);
                drawer.Draw(formattedText, (BitmapImage)FindResource(IronTexture));
            }
        }

        /// <summary>
        /// Returns the pixel offsets for the given case's indexes.
        /// </summary>
        /// <param name="i"></param>
        /// <param name="j"></param>
        /// <returns></returns>
        private Tuple<int, int> CaseOffset(int i, int j)
        {
            return new Tuple<int, int>(i * CaseWidth, j * CaseHeight);
        }

        /// <summary>
        /// Determines the indexes of the top left case according to the current offset.
        /// </summary>
        /// <param name="offsetX">The current pixel X offset</param>
        /// <param name="offsetY">The current pixel Y offset</param>
        /// <returns>A tuple of indices for the map array corresponding to the origin case.</returns>
        private Tuple<int, int> TopLeftCaseIndexes(int offsetX, int offsetY)
        {
            return new Tuple<int, int>(offsetX / CaseWidth,
                                       offsetY / CaseHeight);
        }

        private Tuple<int, int> RelativePositions()
        {
            return new Tuple<int, int>((int)((double)OffsetX * 100 / (Width - (CaseCountX * CaseWidth))),
                                       (int)((double)OffsetY * 100 / (Height - (CaseCountY * CaseHeight))));
        }
        
        #endregion

        #region events
        /// <summary>
        /// Handles mouse clicks. When we click on a case,
        /// we want to display some information about its content.
        /// </summary>
        /// <param name="e"></param>
        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            double x = e.GetPosition(this).X;
            double y = e.GetPosition(this).Y;
            Case c = CaseAtPosition(x + OffsetX, y + OffsetY);
            System.Diagnostics.Debug.WriteLine("Case:" + c.X + "," + c.Y);
        }

        /// <summary>
        /// Handles key inputs. Mainly used to move the visible portion of the map.
        /// </summary>
        /// <param name="e"></param>
        protected override void OnKeyDown(KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Left:
                    if (CanMoveVisibleMapLeft())
                    {
                        MoveVisibleMapLeft();
                    }
                    break;
                case Key.Right:
                    
                    if (CanMoveVisibleMapRight())
                    {
                        MoveVisibleMapRight();
                    }
                    break;
                case Key.Up:
                    if (CanMoveVisibleMapUp())
                    {
                        MoveVisibleMapUp();
                    }
                    break;
                case Key.Down:
                    if (CanMoveVisibleMapDown())
                    {
                        MoveVisibleMapDown();
                    }
                    break;
                default:
                    break;
            }
        }

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

        #region PartialMap
        private bool CanMoveVisibleMapLeft()
        {
            return OffsetX > 0;
        }

        private bool CanMoveVisibleMapRight()
        {
            return OffsetX + (CaseCountX * CaseWidth) <= Width - CaseWidth;
        }

        private bool CanMoveVisibleMapUp()
        {
            return OffsetY > 0;
        }

        private bool CanMoveVisibleMapDown()
        {
            return OffsetY + (CaseCountY * CaseHeight) <= Height - CaseHeight;
        }

        private void MoveVisibleMapLeft()
        {
            OffsetX -= MoveOffset;
            InvalidateVisual();
        }

        private void MoveVisibleMapRight()
        {
            OffsetX += MoveOffset;
            InvalidateVisual();
        }

        private void MoveVisibleMapUp()
        {
            OffsetY -= MoveOffset;
            InvalidateVisual();
        }

        private void MoveVisibleMapDown()
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
