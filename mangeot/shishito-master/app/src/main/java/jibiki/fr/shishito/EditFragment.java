/* Copyright (C) 2016 Thibaut Le Guilly et Mathieu Mangeot
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

package jibiki.fr.shishito;

import android.content.Context;
import android.graphics.Typeface;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v4.widget.TextViewCompat;
import android.text.InputType;
import android.text.TextUtils;
import android.util.Pair;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.w3c.dom.Node;

import java.util.ArrayList;

import jibiki.fr.shishito.Interfaces.OnEntryUpdatedListener;
import jibiki.fr.shishito.Models.Example;
import jibiki.fr.shishito.Models.GramBlock;
import jibiki.fr.shishito.Models.ListEntry;
import jibiki.fr.shishito.Tasks.UpdateContribution;
import jibiki.fr.shishito.Util.ViewUtil;
import jibiki.fr.shishito.Util.XMLUtils;

public class EditFragment extends Fragment implements UpdateContribution.ContributionUpdatedListener {

    @SuppressWarnings("unused")
    private static final String TAG = EditFragment.class.getSimpleName();

    private static final String ENTRY = "entry";

    private static final int KANJI = 100;
    private static final int HIRAGANA = KANJI + 1;
    private static final int ROMAJI_DISPLAY = HIRAGANA + 1;
    private static final int ROMAJI_SEARCH = ROMAJI_DISPLAY + 1;
    private static final int SENS = 500;
    private static final int EXAMPLE_HIRAGANA = 1000;
    private static final int EXAMPLE_ROMAJI = 1500;
    private static final int EXAMPLE_FRENCH = 2000;
    private static final int EXAMPLE_KANJI = 2500;


    private ListEntry entry;

    private OnEntryUpdatedListener mListener;

    private ArrayList<Pair<String, String>> modifWaitList;

    public EditFragment() {
    }

    public static EditFragment newInstance(ListEntry entry) {
        EditFragment fragment = new EditFragment();
        Bundle args = new Bundle();
        args.putSerializable(ENTRY, entry);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            entry = (ListEntry) getArguments().getSerializable(ENTRY);
        }
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        if (context instanceof OnEntryUpdatedListener) {
            mListener = (OnEntryUpdatedListener) context;
        } else {
            throw new RuntimeException(context.toString()
                    + " must implement OnFragmentInteractionListener");
        }
    }

    @Override
    public void onDetach() {
        super.onDetach();
        mListener = null;
    }

    private void addFieldVerif(LinearLayout ll, boolean verif, String text, String title, int id) {
        LinearLayout.LayoutParams params1 = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        TextView tv;
        if (verif) {
            tv = new TextView(getContext());
            TextViewCompat.setTextAppearance(tv, android.R.style.TextAppearance_Medium);
            params1.setMargins(15, 0, 0, 0);
            if (TextUtils.isEmpty(text)) {
                text = getString(R.string.empty_field);
                tv.setTypeface(null, Typeface.ITALIC);
            }
        } else {
            tv = new EditText(getContext());
        }
        tv.setId(id);
        tv.setText(ViewUtil.removeFancy(text));
        tv.setLayoutParams(params1);
        addTitleView(title, ll);
        ll.addView(tv);
    }


    private void addTitleView(String title, LinearLayout ll) {
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        TextView titleView = new TextView(getContext());
        titleView.setText(title);
        titleView.setLayoutParams(titleParams);
        TextViewCompat.setTextAppearance(titleView, android.R.style.TextAppearance_Medium);
        ll.addView(titleView);
    }

    private void addEditView(String content, LinearLayout ll, int id) {
        LinearLayout.LayoutParams editParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        EditText et = new EditText(getContext());
        et.setLayoutParams(editParams);
        et.setText(ViewUtil.removeFancy(content));
        et.setId(id);
        et.setInputType(InputType.TYPE_CLASS_TEXT);
        ll.addView(et);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        final View v = inflater.inflate(R.layout.activity_edit, container, false);
        LinearLayout ll = (LinearLayout) v.findViewById(R.id.editlinear);
        addFieldVerif(ll, entry.isVerified(), XMLUtils.getStringFromNode(entry.getKanjiNode()), getString(R.string.kanji), KANJI);
        //Setting to true ensure that they are not editable... little hack
        addFieldVerif(ll, true, XMLUtils.getStringFromNode(entry.getHiraganaNode()), getString(R.string.hiragana), HIRAGANA);
        addFieldVerif(ll, true, entry.getRomajiDisplay(), getString(R.string.romaji), ROMAJI_DISPLAY);
        addFieldVerif(ll, true, entry.getRomajiSearch(), getString(R.string.romaji_search), ROMAJI_SEARCH);

        int cnt = 1;
        for (GramBlock gram : entry.getGramBlocks()) {
            addTitleView("[" + gram.getGram() + "]", ll);
            int i = 1;
            for (String sense : gram.getSens()) {
                addTitleView("Sens " + i + ":", ll);
                addEditView(sense, ll, SENS + cnt);
                i++;
                cnt++;
            }
        }

        cnt = 1;
        for (Example ex : entry.getExamples()) {
            addTitleView(getString(R.string.edit_example, cnt), ll);
            addTitleView(getString(R.string.edit_example_kanji), ll);
            addEditView(ex.getKanji(), ll, EXAMPLE_KANJI + cnt);
            addTitleView(getString(R.string.edit_example_hiragana), ll);
            addEditView(ex.getHiragana(), ll, EXAMPLE_HIRAGANA + cnt);
            addTitleView(getString(R.string.edit_example_romaji), ll);
            addEditView(ex.getRomaji(), ll, EXAMPLE_ROMAJI + cnt);
            addTitleView(getString(R.string.edit_example_french), ll);
            addEditView(ex.getFrench(), ll, EXAMPLE_FRENCH + cnt);
            cnt++;
        }

        Button saveButton = (Button) v.findViewById(R.id.button);
        saveButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View but) {
                ArrayList<Pair<String, String>> xpaths = new ArrayList<>(4);
                if (!entry.isVerified()) {
                    checkAddPairToArrayList(xpaths, entry.getKanjiNode(), KANJI, v);
                    checkAddPairToArrayList(xpaths, entry.getHiraganaNode(), HIRAGANA, v);
                    //checkAddPairToArrayList(xpaths, entry.getRomajiDisplay(), xpath, ROMAJI_DISPLAY, v);
                    //checkAddPairToArrayList(xpaths, entry.getRomajiSearch(), xpath, ROMAJI_SEARCH, v);
                }

/*
                int i = 1;
                for (GramBlock gramBlock : entry.getGramBlocks()) {
                    for (String sense: gramBlock.getSens()) {
                        String xpath = XMLUtils.getTransformedNumberedXpath(volume, "cdm-definition", "sens", i);
                        checkAddPairToArrayList(xpaths, sense, xpath, SENS + i, v);
                        i++;
                    }
                }

                i = 1;
                for (Example ex : entry.getExamples()) {
                    String xpath = XMLUtils.getTransformedNumberedXpath(volume, "cdm-example-jpn", "exemple", i);
                    checkAddPairToArrayList(xpaths, ex.getKanji(), xpath, EXAMPLE_KANJI + i, v);
                    xpath = XMLUtils.getTransformedNumberedXpath(volume, "cesselin-example-hiragana", "exemple", i);
                    checkAddPairToArrayList(xpaths, ex.getHiragana(), xpath, EXAMPLE_HIRAGANA + i, v);
                    xpath = XMLUtils.getTransformedNumberedXpath(volume, "cesselin-example-romaji", "exemple", i);
                    checkAddPairToArrayList(xpaths, ex.getRomaji(), xpath, EXAMPLE_ROMAJI + i, v);
                    xpath = XMLUtils.getTransformedNumberedXpath(volume, "cdm-example-fra", "exemple", i);
                    checkAddPairToArrayList(xpaths, ex.getFrench(), xpath, EXAMPLE_FRENCH + i, v);
                    i++;
                }
*/


                if (xpaths.size() == 0) {
                    makeToast("Pas de changement détecté.");
                } else { //if (xpaths.size() == 1) {
                    EditFragment.this.modifWaitList = xpaths;
                    EditFragment.this.doNextModif();
                }
//                else {
//                    new PrepareUpdateEntry().execute(xpaths);
//                }

            }
        });

        return v;
    }

    private void doNextModif() {
        Pair<String, String> p = this.modifWaitList.get(0);
        this.modifWaitList.remove(0);
        String xpath;
        String update;
        xpath = p.first;
        update = p.second;

        String[] params = {entry.getContribId(), update, xpath};
        new UpdateContribution(this, entry.getVolume()).execute(params);
    }

    private void checkAddPairToArrayList(ArrayList<Pair<String, String>> a, Node theNode, int id, View v) {
        String value = XMLUtils.getStringFromNode(theNode);
        value = ViewUtil.removeFancy(value);
        if (!value.equals(((EditText) v.findViewById(id)).getText().toString())) {
            String update = ((EditText) v.findViewById(id)).getText().toString();
            String xpath = XMLUtils.getFullXPath(theNode);
            a.add(new Pair<>(xpath, update));
        }
    }

    private void onEntryModified(ListEntry entry) {
        this.mListener.onEntryUpdatedListener(entry);
    }

    private void handleListEntry(ListEntry entry) {
        EditFragment.this.onEntryModified(entry);
    }

    private void saveError() {
        makeToast("Erreur, les modifications n'ont pas été sauvegardés.");
    }

    private void makeToast(String msg) {
        Toast t = Toast.makeText(getContext(), msg, Toast.LENGTH_LONG);
        t.setGravity(Gravity.TOP, 0, 10);
        t.show();
    }

    @Override
    public void onContributionUpdated(ListEntry entry) {
        if (entry != null) {
            if (EditFragment.this.modifWaitList.size() == 0) {
                handleListEntry(entry);
            } else {
                EditFragment.this.entry = entry;
                EditFragment.this.doNextModif();
            }
        } else {
            getActivity().runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    saveError();
                }
            });
        }
    }

/**Keep this part in case want to change back to update entire entry**/
//    private class PrepareUpdateEntry extends AsyncTask<ArrayList<Pair<String, String>>, Void, String> {
//        @Override
//        protected String doInBackground(ArrayList<Pair<String, String>>... params) {
//            String url = SearchActivity.VOLUME_API_URL + EditFragment.this.entry.getContribId() + "/";
//            InputStream stream = HTTPUtils.doGet(url);
//            String res;
//            try {
//                res = XMLUtils.updateEntryXmlFromStream(stream, params[0], ((SearchActivity) getActivity()).getVolume());
//            } catch (ParserConfigurationException | SAXException | IOException | XPathExpressionException e) {
//                Log.e(TAG, "Error updating entry from stream: " + e.getMessage());
//                return null;
//            }
//
//            return res;
//        }
//
//        @Override
//        protected void onPostExecute(String res) {
//            if (res != null) {
//                new UpdateEntry().execute(res);
//            } else {
//                saveError();
//            }
//        }
//    }
//
//    private class UpdateEntry extends AsyncTask<String, Void, ListEntry> {
//
//        @Override
//        protected ListEntry doInBackground(String... params) {
//            String url = SearchActivity.VOLUME_API_URL + EditFragment.this.entry.getEntryId() + "/";
//            InputStream stream = HTTPUtils.doPut(url, params[0]);
//            return handleListEntryStream(stream);
//        }
//
//        @Override
//        protected void onPostExecute(ListEntry entry) {
//            handleListEntry(entry);
//        }
//    }

}
