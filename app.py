import streamlit as st
import pandas as pd
import pickle


def ask_question(attribute_id, description, index):
    options = ["0=Never", "1=Seldom", "2=Averagely", "3=Frequently", "4=Always"]
    answer = st.radio(description, options=options, key=f"{attribute_id}_{index}")
    return int(answer[0])

def main():
    st.title("Divorce prediction")

    questions = [
        (1, "If one of us apologizes when our discussion deteriorates, the discussion ends."),
        (2, "I know we can ignore our differences, even if things get hard sometimes."),
        (3, "When we need it, we can take our discussions with my spouse from the beginning and correct it."),
        (4, "When I discuss with my spouse, to contact him will eventually work."),
        (5, "The time I spent with my wife/husband is special for us."),
        (6, "We don't have time at home as partners."),
        (7, "We are like two strangers who share the same environment at home rather than family."),
        (8, "I enjoy our holidays with my wife/husband."),
        (9, "I enjoy traveling with my wife/husband."),
        (10, "Most of our goals are common to my spouse."),
        (11, "I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other."),
        (12, "My spouse and I have similar values in terms of personal freedom."),
        (13, "My spouse and I have a similar sense of entertainment."),
        (14, "Most of our goals for people (children, friends, etc.) are the same."),
        (15, "Our dreams with my spouse are similar and harmonious."),
        (16, "We're compatible with my spouse about what love should be."),
        (17, "We share the same views about being happy in our life with my spouse."),
        (18, "My spouse and I have similar ideas about how marriage should be."),
        (19, "My spouse and I have similar ideas about how roles should be in marriage."),
        (20, "My spouse and I have similar values in trust."),
        (21, "I know exactly what my wife/husband likes."),
        (22, "I know how my spouse wants to be taken care of when she/he is sick."),
        (23, "I know my spouse's favorite food."),
        (24, "I can tell you what kind of stress my spouse is facing in her/his life."),
        (25, "I have knowledge of my spouse's inner world."),
        (26, "I know my spouse's basic anxieties."),
        (27, "I know what my spouse's current sources of stress are."),
        (28, "I know my spouse's hopes and wishes."),
        (29, "I know my spouse very well."),
        (30, "I know my spouse's friends and their social relationships."),
        (31, "I feel aggressive when I argue with my spouse."),
        (32, "When discussing with my spouse, I usually use expressions such as 'you always' or 'you never'."),
        (33, "I can use negative statements about my spouse's personality during our discussions."),
        (34, "I can use offensive expressions during our discussions."),
        (35, "I can insult my spouse during our discussions."),
        (36, "I can be humiliating when we discuss."),
        (37, "My discussion with my spouse is not calm."),
        (38, "I hate my spouse's way of opening a subject."),
        (39, "Our discussions often occur suddenly."),
        (40, "We're just starting a discussion before I know what's going on."),
        (41, "When I talk to my spouse about something, my calm suddenly breaks."),
        (42, "When I argue with my spouse, I only go out and I don't say a word."),
        (43, "I mostly stay silent to calm the environment a little bit."),
        (44, "Sometimes I think it's good for me to leave home for a while."),
        (45, "I'd rather stay silent than discuss with my spouse."),
        (46, "Even if I'm right in the discussion, I stay silent to hurt my spouse."),
        (47, "When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger."),
        (48, "I feel right in our discussions."),
        (49, "I have nothing to do with what I've been accused of."),
        (50, "I'm not actually the one who's guilty about what I'm accused of."),
        (51, "I'm not the one who's wrong about problems at home."),
        (52, "I wouldn't hesitate to tell my spouse about her/his inadequacy."),
        (53, "When I discuss, I remind my spouse of her/his inadequacy."),
        (54, "I'm not afraid to tell my spouse about her/his incompetence.")
    ]

   
    responses = {}

    for i, (attribute_id, description) in enumerate(questions, 1):
        answer = ask_question(attribute_id, description, i)
        responses[f"Question {attribute_id}"] = answer

    df = pd.DataFrame.from_dict(responses, orient='index', columns=['Response'])
    st.write("Responses:")
    st.dataframe(df.transpose())

    # Load the pickled machine learning model
    pickled_model = pickle.load(open('model.pkl', 'rb'))

    # Prepare the input for prediction
    input_data = pd.Series(responses)

    # Predict the outcome
    prediction = pickled_model.predict([input_data])
    st.write("Using machine learning to predict the outcome")
 
    if prediction[0] == 1:
        st.warning("Based on the survey responses, there is a possibility of relationship issues. Seek professional advice if needed.")
    else:
        st.success("Based on the survey responses, the relationship seems to be stable. Keep nurturing your bond with your spouse.")
    st.info("With the accuracy of 98.3921568627451 % ")


if __name__ == '__main__':
    main()
