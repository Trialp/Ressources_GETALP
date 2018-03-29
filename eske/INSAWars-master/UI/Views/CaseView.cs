using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;
using INSAWars.MVVM;
using System.ComponentModel;
using INSAWars.Units;

namespace UI.Views
{
    /// <summary>
    /// Defines a ViewModel for Case.
    /// </summary>
    public class CaseView : ObservableObject
    {
        private string _occupantName;
        private int _food;
        private int _iron;
        private bool _hasCity;
        private CityView _city;
        private List<UnitView> _units;

        private UnitView _selectedUnitView;
        private bool _selectedUnitCanMove;
        private bool _selectedUnitCanBuildCity;
        private bool _selectedUnitCanAttack;

        private Game _game;
        private Case _case;

        public CaseView(Game g, Case c)
        {
            _game = g;
            _case = c;

            Units = c.Units.Select(u => new UnitView(u)).ToList();
            HasCity = c.HasCity;

            if (c.City != null)
            {
                City = new CityView(c.City);
            }

            OccupantName = (c.Occupant == null) ? "Free" : c.Occupant.Name.ToString();
            Food = c.Food;
            Iron = c.Iron;
            SelectedUnitCanMove = false;
            SelectedUnitCanBuildCity = false;
            SelectedUnitCanAttack = false;
            SelectedUnitView = null;

            c.PropertyChanged += new PropertyChangedEventHandler(delegate(object sender, PropertyChangedEventArgs args)
            {
                var updatedCase = (Case)sender;

                switch (args.PropertyName)
                {
                    case "Units":
                        Units = updatedCase.Units.Select(u => new UnitView(u)).ToList();
                        
                        break;
                    case "HasCity":
                        HasCity = updatedCase.HasCity;
                        break;
                    case "Occupant":
                        OccupantName = (updatedCase.Occupant == null) ? "Free" : updatedCase.Occupant.Name.ToString();
                        break;
                    case "City":
                        // The new city could be null, we need to update only if it's not null
                        if (updatedCase.City != null)
                        {
                            City = new CityView(updatedCase.City);
                        }
                        break;
                    default:
                        break;
                }
            });
        }

        public Case Case
        {
            get { return _case; }
        }

        public string OccupantName
        {
            get { return _occupantName; }
            set
            {
                SetProperty(ref _occupantName, value);
            }
        }

        public CityView City
        {
            get { return _city; }
            set
            {
                SetProperty(ref _city, value);
            }
        }

        public int Food
        {
            get { return _food; }
            set
            {
                SetProperty(ref _food, value);
            }
        }

        public int Iron
        {
            get { return _iron; }
            set
            {
                SetProperty(ref _iron, value);
            }
        }

        public bool HasCity
        {
            get { return _hasCity; }
            set
            {
                SetProperty(ref _hasCity, value);
            }
        }

        public List<UnitView> Units
        {
            get { return _units; }
            set
            {
                SetProperty(ref _units, value);
            }
        }

        public UnitView SelectedUnitView
        {
            get { return _selectedUnitView; }
            set
            {
                SetProperty(ref _selectedUnitView, value);
                SelectedUnitCanMove = (value != null) && (value.Unit.Player == _game.CurrentPlayer) && value.Unit.CanMove();
                SelectedUnitCanBuildCity = (value != null) && (value.Unit.Player == _game.CurrentPlayer) && value.Unit is Teacher;
                SelectedUnitCanAttack = (value != null) && (value.Unit.Player == _game.CurrentPlayer) && value.Unit.CanAttack();
            }
        }

        public bool SelectedUnitCanMove
        {
            get { return _selectedUnitCanMove; }
            set
            {
                SetProperty(ref _selectedUnitCanMove, value);
            }
        }

        public bool SelectedUnitCanBuildCity
        {
            get { return _selectedUnitCanBuildCity; }
            set
            {
                SetProperty(ref _selectedUnitCanBuildCity, value);
            }
        }

        public bool SelectedUnitCanAttack
        {
            get { return _selectedUnitCanAttack; }
            set
            {
                SetProperty(ref _selectedUnitCanAttack, value);
            }
        }
    }
}
