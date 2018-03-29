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
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace UI.Drawing
{
    /// <summary>
    /// Provides utilities to draw icons and text on a case.
    /// Draws up to 4 inline items and 1 main item.  
    /// </summary>
    public class CaseDrawer
    {
        private const int ItemWidth = 48;
        private const int ItemHeight = 20;

        private int _itemCount;
        private DrawingContext _context;
        private Tuple<int, int> _origin;

        public CaseDrawer(DrawingContext context, Tuple<int, int> origin)
        {
            _itemCount = 0;
            _context = context;
            _origin = origin;
        }

        /// <summary>
        /// Draws the given icon (and text if provided) on the next line.
        /// </summary>
        /// <param name="icon"></param>
        /// <param name="text"></param>
        public void Draw(BitmapImage icon, FormattedText text=null)
        {
            if (_itemCount < 4)
            {
                DrawLeft(icon, text);
                _itemCount++;
            }     
        }

        private void DrawLeft(BitmapImage icon, FormattedText text = null)
        {
            _context.DrawImage(icon, new Rect(_origin.Item1 + 8,
                                                    _origin.Item2 + 8 + _itemCount * ItemHeight,
                                                    16, 16));
            if (text != null)
            {
                _context.DrawText(text, new Point(_origin.Item1 + 28, _origin.Item2 + 8 + _itemCount * ItemHeight));
            }
        }

        /// <summary>
        /// Draws an icon on the entire case.
        /// </summary>
        /// <param name="icon"></param>
        public void DrawMainItem(BitmapImage icon)
        {
            _context.DrawImage(icon, new Rect(_origin.Item1,
                                                    _origin.Item2,
                                                    96, 96));
        }

        /// <summary>
        /// Resets the number of items.
        /// </summary>
        public void Clear()
        {
            _itemCount = 0;
        }
   
    }
}
