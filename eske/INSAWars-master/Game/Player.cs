#region usings
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Units;
using INSAWars.MVVM;
using System.ComponentModel;
#endregion

namespace INSAWars.Game
{
    [Serializable]
    public class Player : ObservableObject
    {
        #region fields
        private bool isDead;
        private List<City> cities;
        private ICivilization civilization;
        private string name;
        private List<Unit> units;
        #endregion

        #region properties
        public List<City> Cities
        {
            get { return cities; }
        }

        public int CitiesCount
        {
            get { return Cities.Count(); }
        }

        public List<Unit> Units
        {
            get { return units; }
        }

        public Head Head { get; set; }

        public bool HasHead
        {
            get { return Head != null; }
        }

        public bool IsDead
        {
            get { return isDead; }
        }

        public string Name
        {
            get { return name; }
            set
            {
                SetProperty(ref name, value);
            }
        }

        public ICivilization Civilization
        {
            get { return civilization; }
        }
        #endregion

        #region constructors
        public Player(ICivilization civilization, string name)
        {
            isDead = false;
            cities = new List<City>();
            units = new List<Unit>();
            this.civilization = civilization;
            Name = name;
            Head = null;
            OnPropertyChanged("CitiesCount");
            OnPropertyChanged("Units");
        }
        #endregion

        #region methods
        /// <summary>
        /// Adds a city.
        /// </summary>
        /// <param name="city"></param>
        public void AddCity(City city)
        {
            cities.Add(city);
            OnPropertyChanged("CitiesCount");
        }

        /// <summary>
        /// Removes a city.
        /// </summary>
        /// <param name="city"></param>
        public void RemoveCity(City city)
        {
            cities.Remove(city);
            OnPropertyChanged("CitiesCount");

            if (LoseCondition())
            {
                Lose();
            }
        }

        /// <summary>
        /// Adds a unit.
        /// </summary>
        /// <param name="unit"></param>
        public void AddUnit(Unit unit)
        {
            units.Add(unit);
            OnPropertyChanged("Units");
        }

        /// <summary>
        /// Removes a unit.
        /// </summary>
        /// <param name="unit"></param>
        public void RemoveUnit(Unit unit)
        {
            units.Remove(unit);
            OnPropertyChanged("Units");
            if (LoseCondition())
            {
                Lose();
            }
        }

        /// <summary>
        /// Returns true if the player has no way to get back into the game, and thus has lost.
        /// A player has lost if he has no city and no teacher left alive.
        /// </summary>
        /// <returns></returns>
        private bool LoseCondition()
        {
            return cities.Count == 0 && !units.ToList().Exists(item => item is Teacher);
        }

        /// <summary>
        /// Kills the player and its units and destroys its cities.
        /// </summary>
        public void Lose()
        {
            isDead = true;

            // If the player has surrendered
            foreach (City city in new List<City>(cities))
            {
                city.Destroy();
            }

            foreach (Unit unit in new List<Unit>(units))
            {
                unit.Kill();
            }
        }

        /// <summary>
        /// Updates the player's cities and units.
        /// </summary>
        public void NextTurn()
        {
            foreach (City city in cities)
            {
                city.NextTurn();
            }

            foreach (Unit unit in units)
            {
                unit.Reset();
            }
        }

        public override bool Equals(object obj)
        {
            Player playerObj = obj as Player;

            return playerObj != null && name == playerObj.name;
        }

        public override int GetHashCode()
        {
            return name.GetHashCode();
        }

        public override string ToString()
        {
            return "Player " + name + " (" + (isDead ? "dead" : "alive") + ")";
        }
        #endregion
    }
}
