package com.adap;

import com.krishna.PilotPen;

public class School {
    PilotPen pp = new PilotPen();
    Pen p = new PenAdapter(pp);
    AssignmentWork aw = new AssignmentWork();
    aw.setP(p);
    aw.writeAssignment();
}
