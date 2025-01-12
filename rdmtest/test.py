from graphviz import Digraph

# Initialize the UML diagram
uml_diagram = Digraph('UML_Class_Diagram', filename='./UML_Diagram_Complete.png', format='png')
uml_diagram.attr(rankdir='LR', size='8,5')

# Classes with attributes and methods for the provided code
classes = {
    "AppointmentView": {
        "attributes": ["scanner: Scanner", "appointmentController: AppointmentController", 
                       "patientView: PatientView", "dermatologistView: DermatologistView", 
                       "treatmentView: TreatmentView", "timeView: TimeView", "invoiceView: InvoiceView"],
        "methods": ["showAppointmentMenu(): void", "displayMenu(): void", "getUserChoice(): int", 
                    "scheduleAppointment(): void", "viewAllAppointments(): void", 
                    "updateAppointment(): void", "cancelAppointment(): void", 
                    "completeAppointment(): void", "processInvoicePayment(): void", 
                    "searchAppointment(): void"]
    },
    "AppointmentController": {
        "attributes": [],
        "methods": ["createAppointment(patient, dermatologist, treatment, time, invoice): Appointment", 
                    "listAppointments(): List<Appointment>", "getAppointmentById(appointmentId: int): Appointment", 
                    "updateAppointment(appointmentId: int, patient, dermatologist, treatment, time): bool", 
                    "cancelAppointmentById(appointmentId: int): bool"]
    },
    "PatientView": {
        "attributes": [],
        "methods": ["getPatientDetails(): Patient"]
    },
    "DermatologistView": {
        "attributes": ["scanner: Scanner", "dermatologistRepository: DermatologistRepository"],
        "methods": ["selectDermatologist(): DermatologistModel", 
                    "addTimeSlotForDermatologist(): void", "getValidDateFromUser(): LocalDate", 
                    "getValidStartTimeFromUser(): LocalTime", "getValidInputFromUser(prompt, parser): <T>"]
    },
    "TreatmentView": {
        "attributes": [],
        "methods": ["getTreatmentDetails(): Treatment"]
    },
    "TimeView": {
        "attributes": [],
        "methods": ["getTimeDetails(dermatologist): Time"]
    },
    "InvoiceView": {
        "attributes": [],
        "methods": ["getInvoiceDetails(treatment): Invoice"]
    },
    "Appointment": {
        "attributes": ["appointmentId: int", "patient: Patient", "dermatologist: DermatologistModel", 
                       "treatment: Treatment", "time: Time", "invoice: Invoice", "status: String"],
        "methods": []
    },
    "Patient": {
        "attributes": ["patientId: int", "name: String", "age: int", "gender: String"],
        "methods": []
    },
    "DermatologistModel": {
        "attributes": ["dermatologistId: int", "name: String", "specialization: String"],
        "methods": ["checkAvailability(time: Time): bool"]
    },
    "Treatment": {
        "attributes": ["treatmentId: int", "name: String", "description: String", "cost: double"],
        "methods": []
    },
    "Time": {
        "attributes": ["date: LocalDate", "startTime: LocalTime", "endTime: LocalTime"],
        "methods": []
    },
    "Invoice": {
        "attributes": ["invoiceId: int", "totalAmount: double", "amountPaid: double", "isPaid: bool"],
        "methods": ["makePayment(amount: double): void", "isPaid(): bool"]
    },
    "Dermatologist": {
        "attributes": ["name: String"],
        "methods": []
    },
    "TimeImpl": {
        "attributes": ["date: LocalDate", "startTime: LocalTime", "endTime: LocalTime"],
        "methods": ["getEndTime(): LocalTime"]
    },
    "DermatologistRepository": {
        "attributes": [],
        "methods": ["addTimeSlot(dermatologist: DermatologistModel, time: Time): bool"]
    }
}

# Adding classes with their attributes and methods to the UML
for class_name, class_info in classes.items():
    attributes = "\l".join(class_info["attributes"]) + "\l"
    methods = "\l".join(class_info["methods"]) + "\l"
    uml_diagram.node(class_name, f"{class_name}|\l{attributes}|{methods}")

# Relationships among classes (associations, dependencies, etc.)
relationships = [
    ("AppointmentView", "AppointmentController", "uses"),
    ("AppointmentView", "PatientView", "uses"),
    ("AppointmentView", "DermatologistView", "uses"),
    ("AppointmentView", "TreatmentView", "uses"),
    ("AppointmentView", "TimeView", "uses"),
    ("AppointmentView", "InvoiceView", "uses"),
    ("Appointment", "Patient", "aggregation"),
    ("Appointment", "DermatologistModel", "aggregation"),
    ("Appointment", "Treatment", "aggregation"),
    ("Appointment", "Time", "aggregation"),
    ("Appointment", "Invoice", "aggregation"),
    ("DermatologistModel", "Dermatologist", "composition"),
    ("DermatologistView", "DermatologistRepository", "uses"),
    ("DermatologistRepository", "DermatologistModel", "aggregation"),
    ("TimeImpl", "Time", "implements"),
]

# Adding relationships with their types (composition, aggregation, etc.) to the UML
for src, dst, relationship_type in relationships:
    if relationship_type == "composition":
        uml_diagram.edge(src, dst, arrowhead="diamond")
    elif relationship_type == "aggregation":
        uml_diagram.edge(src, dst, arrowhead="odiamond")
    else:
        uml_diagram.edge(src, dst)

uml_diagram.attr(label="Complete UML Class Diagram for Appointment Management System", fontsize="20")
uml_diagram.attr(labelloc="t")

uml_diagram.attr(dpi='300')  # Set the DPI to 300 for better resolution
uml_diagram.attr(rankdir='LR', size='12,8')  # Increase the diagram size for more detail

uml_diagram.render(filename='./UML_Diagram_Complete.png', format='png', cleanup=False)


