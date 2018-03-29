using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace INSAWars.MVVM
{
    /// <summary>
    /// This class provides a common implementation of INotifyPropertyChanged, in order to
    /// (partially) implement the MVVM pattern.
    /// </summary>
    [Serializable]
    public abstract class ObservableObject : INotifyPropertyChanged
    {
        [field: NonSerialized]
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// Notifies listeners that the property has changed and sets the property to its new value.
        /// This solution is based on a new C#5 Attribute, [CallerMemberName], that allows us
        /// to avoid the use of a string to get the property name.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="field"></param>
        /// <param name="value"></param>
        /// <param name="propName"></param>
        protected void SetProperty<T>(ref T field, T value, [CallerMemberName] string propName = null)
        {
            if (!EqualityComparer<T>.Default.Equals(field, value))
            {
                field = value;
                var pc = PropertyChanged;
                if (pc != null)
                {
                    pc(this, new PropertyChangedEventArgs(propName));
                }
            }
        }

        /// <summary>
        /// Notifies listeners that the property has changed. Does not sets a new value for the property.
        /// This is useful for properties not backed by a private variable, but is a bit dirtier than the
        /// more modern C#5 version.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="propName"></param>
        protected void OnPropertyChanged(string propName)
        {
            var pc = PropertyChanged;
            if (pc != null)
            {
                pc(this, new PropertyChangedEventArgs(propName));
            }
        }

    }
}
