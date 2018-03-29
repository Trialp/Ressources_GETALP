using System;
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
using INSAWars.Game;

namespace UI
{
    /// <summary>
    /// Provides controls to let the user choose the map's size.
    /// </summary>
    public partial class ChooseMapSizePage : Page
    {
        private GameBuilder _builder;

        public ChooseMapSizePage(GameBuilder builder)
        {
            _builder = builder;

            InitializeComponent();
        }

        /// <summary>
        /// Go to the previous page.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BackButtonClick(object sender, RoutedEventArgs e)
        {
            NavigationService.GoBack();
        }

        /// <summary>
        /// Creates the game's window.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void NextButtonClick(object sender, RoutedEventArgs e)
        {
            _builder.SetSize(_mapSize.Text);
            _builder.UseDefaultFrequencies();

            var main = Application.Current.MainWindow;
            var game = new GameWindow(_builder.Build());            
            Application.Current.MainWindow = game;
            main.Close();
            game.Show();
        }
    }
}
