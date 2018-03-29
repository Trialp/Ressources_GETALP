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
    /// The in-game menu provides actions to create a new game, save the curent game or exit the application.
    /// </summary>
    public partial class InGameMenuWindow : NavigationWindow
    {
        private Game _game;

        public InGameMenuWindow(Game currentGame)
        {
            _game = currentGame;
            InitializeComponent();
        }

        /// <summary>
        /// Hides the menu if the player pushes Escape.
        /// </summary>
        /// <param name="e"></param>
        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                Close();
            }
        }

        /// <summary>
        /// Transfers the current game to the home page.
        /// This is necessary since a Game instance is needed to save it to a file.
        /// Since InGameMenuWindow is a NavigationWindow, this method with be called everytime a
        /// new page is loaded. This is why we need to check that the home page was loaded.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LoadComplete(object sender, NavigationEventArgs e)
        {
            var homePage = e.Content as InGameMenuHomePage;
            if (homePage != null)
            {
                homePage.CurrentGame = _game;
            }           
        }
    }
}
